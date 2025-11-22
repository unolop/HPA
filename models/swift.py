from typing import List, Optional, Dict, Any, Tuple
import torch
import os 
from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput, GenerateConfig
from evalscope.api.tool import ToolChoice, ToolInfo

# SWIFT/PtEngine related libraries
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

from evalscope.api.model.model import ModelAPI

import base64
import tempfile

def _save_base64_to_temp_file(base64_data: str) -> Optional[str]:
    """Decodes Base64 data (e.g., data:image/jpeg;base64,...) and saves it to a temporary file."""
    try:
        # Extract the actual Base64 string after the comma (if present)
        if ',' in base64_data:
            _, encoded_data = base64_data.split(',', 1)
        else:
            encoded_data = base64_data
        
        image_bytes = base64.b64decode(encoded_data)
        
        # Create a temporary file and write the image bytes
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(image_bytes)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        return None


class SwiftVLMEngine(ModelAPI):
    """Evalscope ModelAPI implementation using MS-SWIFT PtEngine for VLM inference."""
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        custom_output: Optional[str] = None,
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        self.custom_output = custom_output

        # 1. Load Model and Tokenizer
        self.model, self.tokenizer = get_model_tokenizer(model_name, use_hf=True, max_pixels=448)
        self.model.eval()
        
        # 2. Load Template and PtEngine
        template_type = self.model.model_meta.template
        template = get_template(template_type, self.tokenizer, default_system=None)
        self.engine = PtEngine.from_model_template(self.model, template, max_batch_size=1)
    
    def _process_messages(self, messages: List[ChatMessage]) -> Tuple[str, List[str]]:
        """
        Processes ChatMessage, handling evalscope's ContentText and ContentImage objects.
        
        Evalscope uses custom content objects:
        - ContentText: has 'type' and 'text' attributes
        - ContentImage: has 'type' and 'image' attributes
        """
        user_message = messages[-1]
        input_text = ""
        image_paths = []

        # Handle string content
        if not isinstance(user_message.content, list):
             input_text = str(user_message.content)
             return input_text, image_paths
        
        for part in user_message.content:
            # Get the type attribute (works for both dict and objects)
            part_type = getattr(part, 'type', None) or (part.get('type') if isinstance(part, dict) else None)
            
            # 1. Text handling (ContentText object or dict)
            if part_type == 'text':
                text_content = (
                    getattr(part, 'text', None) or 
                    getattr(part, 'content', None) or 
                    (part.get('text') if isinstance(part, dict) else None) or
                    (part.get('content') if isinstance(part, dict) else None) or
                    ''
                )
                if text_content:
                    input_text += text_content
            
            # 2. Image handling (ContentImage object or dict)
            elif part_type == 'image':
                image_data = (
                    getattr(part, 'image', None) or
                    getattr(part, 'content', None) or
                    (part.get('image') if isinstance(part, dict) else None) or
                    (part.get('content') if isinstance(part, dict) else None)
                )
                
                if image_data and isinstance(image_data, str):
                    # Handle Base64 encoded images
                    if image_data.startswith('data:image') or 'base64' in image_data:
                        temp_path = _save_base64_to_temp_file(image_data)
                        if temp_path:
                            image_paths.append(temp_path)
                    
                    # Handle local file paths
                    elif image_data.startswith('local:'):
                        local_path = image_data[6:]
                        if os.path.exists(local_path):
                            image_paths.append(local_path)
                    
                    # Handle direct file paths
                    elif os.path.exists(image_data):
                        image_paths.append(image_data)
            
            # 3. image_url handling
            elif part_type == 'image_url':
                image_url_data = getattr(part, 'image_url', None) or (part.get('image_url') if isinstance(part, dict) else None)
                
                if image_url_data:
                    url = getattr(image_url_data, 'url', None) or (image_url_data.get('url') if isinstance(image_url_data, dict) else None)
                    
                    if url:
                        if url.startswith('data:image') or 'base64' in url:
                            temp_path = _save_base64_to_temp_file(url)
                            if temp_path:
                                image_paths.append(temp_path)
                        elif url.startswith('local:'):
                            local_path = url[6:]
                            if os.path.exists(local_path):
                                image_paths.append(local_path)
                        elif os.path.exists(url):
                            image_paths.append(url)

        return input_text, image_paths

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        
        # Process Input Messages
        input_text, image_paths = self._process_messages(input)
        
        if not input_text and not image_paths:
            return ModelOutput.from_content(model=self.model_name, content="")
            
        # Create InferRequest
        infer_request = InferRequest(
            messages=[{'role': 'user', 'content': input_text}],
            images=image_paths
        )

        # Create RequestConfig
        request_config = RequestConfig(
            max_tokens=config.max_tokens, 
            temperature=config.temperature,
            top_p=config.top_p,
        )
        
        # Call PtEngine Inference
        with torch.no_grad():
            resp_list = self.engine.infer([infer_request], request_config)
            output_text = resp_list[0].choices[0].message.content
        
        # Return Standardized Output
        return ModelOutput.from_content(
            model=self.model_name,
            content=output_text
        )