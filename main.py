from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union, 
    TypeVar, 
    Protocol,
    runtime_checkable,
    Final,
    cast,
    Any
)
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput
import numpy as np
from numpy.typing import NDArray
import logging
from PIL import Image
import json
from datetime import datetime
from rich.logging import RichHandler
import wandb
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypeAlias

# Type definitions
ImagePath: TypeAlias = Path
ImageTensor: TypeAlias = Tensor  # [B, C, H, W]
Probabilities: TypeAlias = Tensor  # [B, num_classes] with sum=1
LogProbabilities: TypeAlias = Tensor  # [B, num_classes]
BatchIdx: TypeAlias = int
ContextSize: TypeAlias = Literal[10]  # Fixed context size from paper
Reward: TypeAlias = Literal[-1, 1]  # Binary rewards only

class Role(str, Enum):
    SPEAKER = "speaker"
    LISTENER = "listener"

class Phase(str, Enum):
    TRAIN = auto()
    EVAL = auto()
    DEPLOY = auto()

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the COGEN model"""
    model_name: str = "HuggingFaceM4/IDEFICS-2-8B"
    lambda_l: float = 0.5
    lambda_s: float = 0.0
    temperature: float = 0.7
    num_samples: int = 10
    max_seq_length: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    batch_size: int = 32
    num_train_epochs: int = 15
    early_stopping_patience: int = 5
    
    def __post_init__(self) -> None:
        # Validate configuration
        if not 0 <= self.lambda_l <= 1:
            raise ValueError("lambda_l must be between 0 and 1")
        if not 0 <= self.lambda_s <= 1:
            raise ValueError("lambda_s must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

@dataclass(frozen=True)
class Interaction:
    """Single interaction during deployment"""
    context: tuple[ImagePath, ...] 
    utterance: str
    target_idx: int
    reward: Reward
    role: Role
    round: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        if len(self.context) != 10:  # type: ignore
            raise ValueError("Context must contain exactly 10 images")
        if not 0 <= self.target_idx < len(self.context):
            raise ValueError("Invalid target index")

@runtime_checkable
class ImageEncoder(Protocol):
    """Protocol for image encoding components"""
    def encode(self, images: Sequence[ImagePath]) -> ImageTensor:
        ...

class IDEFICSImageEncoder:
    """IDEFICS-specific image encoding"""
    def __init__(
        self, 
        processor: Any,
        device: torch.device
    ) -> None:
        self.processor = processor
        self.device = device
        
    def encode(self, images: Sequence[ImagePath]) -> ImageTensor:
        processed_images: List[ImageTensor] = []
        for img_path in images:
            image = Image.open(img_path).convert('RGB')
            processed = self.processor(image, return_tensors="pt")
            processed_images.append(processed.pixel_values)
        return torch.cat(processed_images).to(self.device)

class COGEN:
    """Main COGEN implementation with coupled comprehension and generation"""
    
    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or self._setup_logger()
        
        # Initialize model and tokenizer
        self.model = self._load_model(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.image_encoder = IDEFICSImageEncoder(
            self.model.get_image_processor(), 
            self.device
        )
        
        # Track probabilities for IPS coefficient
        self.round_probs: Dict[int, float] = {}
        
        self.logger.info(
            f"Initialized COGEN model on {self.device} "
            f"with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def comprehend(
        self,
        context: Sequence[ImagePath],
        utterance: str,
        use_joint_inference: bool = True,
    ) -> int:
        """Listener role: Select target image given context and utterance"""
        self.model.eval()
        with torch.no_grad():
            # Base listener distribution P_l(t|I,u)
            P_l: Probabilities = self._compute_listener_distribution(
                context,
                utterance
            )
            
            if not use_joint_inference:
                return int(torch.argmax(P_l).item())
                
            # Joint inference with speaker model
            P_joint = torch.zeros_like(P_l)
            for t in range(len(context)):
                # P_s(u|I,t) from speaker model
                P_s = self._compute_speaker_probability(context, t, utterance)
                # Compute joint distribution
                P_joint[t] = (
                    P_l[t] ** self.config.lambda_l * 
                    P_s ** (1 - self.config.lambda_l)
                )
                
            # Normalize and select most likely target
            P_joint = P_joint / P_joint.sum()
            return int(torch.argmax(P_joint).item())

    def generate(
        self,
        context: Sequence[ImagePath],
        target_idx: int,
        use_joint_inference: bool = True,
    ) -> str:
        """Speaker role: Generate description for target image"""
        self.model.eval()
        with torch.no_grad():
            # Sample k utterances from base speaker distribution P_s(u|I,t)
            utterances: List[str] = []
            scores: List[float] = []
            
            for _ in range(self.config.num_samples):
                utterance = self._sample_utterance(context, target_idx)
                utterances.append(utterance)
                
                if use_joint_inference:
                    # Score with joint distribution
                    P_s = self._compute_speaker_probability(
                        context,
                        target_idx,
                        utterance
                    )
                    P_l = self._compute_listener_distribution(
                        context,
                        utterance
                    )[target_idx]
                    score = float(
                        P_l ** self.config.lambda_s * 
                        P_s ** (1 - self.config.lambda_s)
                    )
                else:
                    # Score with base speaker distribution only
                    score = float(self._compute_speaker_probability(
                        context,
                        target_idx,
                        utterance
                    ))
                scores.append(score)
                
            # Return utterance with highest score
            best_idx = int(np.argmax(scores))
            return utterances[best_idx]

    def train(
        self,
        comprehension_data: Sequence[Interaction],
        generation_data: Sequence[Interaction],
    ) -> None:
        """Train model using REINFORCE with data sharing"""
        self.model.train()
        
        # Expand training data through data sharing
        comp_data = list(comprehension_data)
        gen_data = list(generation_data)
        
        # Convert successful speaker interactions to listener examples
        for interaction in generation_data:
            if interaction.reward == 1:
                comp_data.append(Interaction(
                    context=interaction.context,
                    utterance=interaction.utterance,
                    target_idx=interaction.target_idx,
                    reward=cast(Reward, 1),
                    role=Role.LISTENER,
                    round=interaction.round
                ))
                
        # Convert successful listener interactions to speaker examples
        for interaction in comprehension_data:
            if interaction.reward == 1:
                gen_data.append(Interaction(
                    context=interaction.context,
                    utterance=interaction.utterance,
                    target_idx=interaction.target_idx,
                    reward=cast(Reward, 1),
                    role=Role.SPEAKER,
                    round=interaction.round
                ))

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        for batch in self._get_batches(comp_data + gen_data):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=self.device)
            
            for interaction in batch:
                if interaction.role == Role.LISTENER:
                    # Listener (comprehension) loss
                    P_l = self._compute_listener_distribution(
                        interaction.context,
                        interaction.utterance
                    )
                    c_l = self._compute_ips_coefficient(
                        interaction.reward,
                        float(P_l[interaction.target_idx].item()),
                        interaction.round
                    )
                    loss = loss - c_l * interaction.reward * torch.log(
                        P_l[interaction.target_idx]
                    )
                
                else:
                    # Speaker (generation) loss
                    P_s = self._compute_speaker_probability(
                        interaction.context,
                        interaction.target_idx,
                        interaction.utterance
                    )
                    c_s = self._compute_ips_coefficient(
                        interaction.reward,
                        float(P_s),
                        interaction.round
                    )
                    loss = loss - c_s * interaction.reward * torch.log(P_s)
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

    def _compute_listener_distribution(
        self,
        context: Sequence[ImagePath],
        utterance: str,
    ) -> Probabilities:
        """Compute P_l(t|I,u) distribution over targets"""
        prompt = self._format_listener_prompt(context, utterance)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        image_features = self.image_encoder.encode(context)
        
        outputs: CausalLMOutput = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=image_features,
            return_dict=True
        )
        
        logits: Tensor = outputs.logits[:, -1, :] / self.config.temperature
        return F.softmax(logits, dim=-1)

    def _compute_speaker_probability(
        self,
        context: Sequence[ImagePath],
        target_idx: int,
        utterance: str,
    ) -> float:
        """Compute P_s(u|I,t) probability"""
        prompt = self._format_speaker_prompt(context, target_idx, utterance)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        image_features = self.image_encoder.encode(context)
        
        outputs: CausalLMOutput = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=image_features,
            return_dict=True
        )
        
        logits: Tensor = outputs.logits[:, -1, :] / self.config.temperature
        return float(F.softmax(logits, dim=-1)[0, 0].item())

    def _sample_utterance(
        self,
        context: Sequence[ImagePath],
        target_idx: int,
    ) -> str:
        """Sample utterance from P_s(u|I,t)"""
        prompt = self._format_speaker_prompt(context, target_idx)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        image_features = self.image_encoder.encode(context)
        
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=image_features,
            temperature=self.config.temperature,
            max_length=self.config.max_seq_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

    def _compute_ips_coefficient(
        self,
        reward: Reward,
        prob: float,
        round: int,
    ) -> float:
        """Compute importance sampling coefficient"""
        if reward == -1:
            return min(prob / self.round_probs.get(round, prob), 5.0)
        return 1.0
    
    def _format_speaker_prompt(
        self,
        context: Sequence[ImagePath],
        target_idx: int,
        utterance: Optional[str] = None,
    ) -> str:
        """Format model inputs for generation"""
        prompt = (
            "[User] You will be presented with a sequence of 10 images and be "
            "assigned a target image. Your task is to produce a caption for your "
            "target image such that anyone could guess the image from your description. "
        )
        
        # Add image references
        for i, _ in enumerate(context):
            prompt += f"Image {i}: <image>, "
            
        prompt += f"Your target is Image {target_idx}. "
        
        if utterance is not None:
            prompt += f"Caption: {utterance}"
        else:
            prompt += "Produce your caption now."
            
        prompt += "\n[Assistant]"
        return prompt

    def _get_batches(
        self,
        data: Sequence[Interaction],
        shuffle: bool = True,
    ) -> Iterator[List[Interaction]]:
        """Create batches for training"""
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
        
        for i in range(0, len(data), self.config.batch_size):
            batch_indices = indices[i:min(i + self.config.batch_size, len(data))]
            yield [data[j] for j in batch_indices]

    def _load_model(self, checkpoint_path: Optional[Path] = None) -> PreTrainedModel:
        """Load model from checkpoint or initialize from scratch"""
        if checkpoint_path is not None:
            self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            return AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.logger.info(f"Initializing model from: {self.config.model_name}")
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger("COGEN")
        logger.setLevel(logging.INFO)
        
        # Rich console handler
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        # File handler for detailed logs
        log_path = Path("logs") / f"cogen_{datetime.now():%Y%m%d_%H%M%S}.log"
        log_path.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration and metadata
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_config": self.config.__dict__,
                    "round_probs": self.round_probs,
                    "timestamp": datetime.now().isoformat()
                },
                f,
                indent=2
            )
        
        self.logger.info(f"Saved checkpoint to: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint"""
        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")
            
        self.model = self._load_model(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load configuration and metadata
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                self.config = ModelConfig(**data["model_config"])
                self.round_probs = data["round_probs"]
        
        self.logger.info(f"Loaded checkpoint from: {path}")

    def deploy(
        self,
        contexts: Sequence[Sequence[ImagePath]],
        role: Role,
        use_joint_inference: bool = True,
    ) -> List[Tuple[str, int]]:
        """Deploy model for multiple interactions"""
        self.logger.info(f"Deploying model in {role} role for {len(contexts)} contexts")
        results: List[Tuple[str, int]] = []
        
        for context in contexts:
            try:
                if role == Role.SPEAKER:
                    target_idx = np.random.randint(len(context))
                    utterance = self.generate(context, target_idx, use_joint_inference)
                    results.append((utterance, target_idx))
                else:  # LISTENER
                    target_utterance = None  # Would be provided by human in real deployment
                    if target_utterance is None:
                        raise ValueError("Utterance required for listener role")
                    selected_idx = self.comprehend(context, target_utterance, use_joint_inference)
                    results.append((target_utterance, selected_idx))
            except Exception as e:
                self.logger.error(f"Error in deployment: {str(e)}")
                continue
                
        return results

def create_cogen_model(
    model_name: str = "HuggingFaceM4/IDEFICS-2-8B",
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    **config_kwargs: Any,
) -> COGEN:
    """Factory function to create COGEN model with custom configuration"""
    config = ModelConfig(model_name=model_name, **config_kwargs)
    return COGEN(config, device=device, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    # Example usage
    model = create_cogen_model()
    
    # Example contexts (would load actual image paths)
    contexts = [[Path(f"image_{i}_{j}.jpg") for j in range(10)] for i in range(5)]
    
    # Deploy as speaker
    speaker_results = model.deploy(contexts, Role.SPEAKER)
    for utterance, target_idx in speaker_results:
        print(f"Generated: {utterance} for target {target_idx}")
   