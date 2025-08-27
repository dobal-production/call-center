import boto3
import json
import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BedrockClient:
    """Amazon Bedrock client for AI model interactions"""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize Bedrock client
        
        Args:
            region_name: AWS region name (optional, defaults to environment variable)
        """
        try:
            # Get region from environment variable or use provided value or default
            self.region_name = region_name or os.getenv('AWS_REGION', 'ap-northeast-2')
            
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name
            )
        except Exception as e:
            st.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        model_id: str, 
        inference_config: Optional[dict] = None,
        custom_system_message: Optional[str] = None
    ) -> str:
        """
        Generate response using Amazon Bedrock Converse API (non-streaming)
        
        Args:
            prompt: User input prompt
            model_id: Bedrock model identifier
            inference_config: Dictionary with inference parameters (maxTokens, temperature, topP)
            custom_system_message: Optional custom system message
            
        Returns:
            Generated response text
        """
        try:
            # Default system message for call center assistant
            default_system_message = """You are a helpful call center assistant. 
            Please provide a professional and empathetic response to customer inquiries.
            
            Guidelines:
            - Be polite and professional
            - Provide clear and actionable solutions
            - Show empathy for customer concerns
            - Keep responses concise but comprehensive
            """
            
            # Use custom system message if provided, otherwise use default
            system_message = custom_system_message if custom_system_message else default_system_message
            
            # Prepare messages for Converse API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Default inference configuration
            default_inference_config = {
                "maxTokens": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
            
            # Use provided inference config or default
            final_inference_config = inference_config if inference_config else default_inference_config
            
            # Make the Converse API call
            response = self.bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": system_message}],
                inferenceConfig=final_inference_config
            )
            
            # Extract the response text
            return response['output']['message']['content'][0]['text']
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def generate_response_stream(
        self, 
        prompt: str, 
        model_id: str, 
        inference_config: Optional[dict] = None,
        custom_system_message: Optional[str] = None
    ):
        """
        Generate streaming response using Amazon Bedrock Converse Stream API
        
        Args:
            prompt: User input prompt
            model_id: Bedrock model identifier
            inference_config: Dictionary with inference parameters (maxTokens, temperature, topP)
            custom_system_message: Optional custom system message
            
        Yields:
            Streaming response chunks
        """
        try:
            # Default system message for call center assistant
            default_system_message = """You are a helpful call center assistant. 
            Please provide a professional and empathetic response to customer inquiries.
            
            Guidelines:
            - Be polite and professional
            - Provide clear and actionable solutions
            - Show empathy for customer concerns
            - Keep responses concise but comprehensive
            """
            
            # Use custom system message if provided, otherwise use default
            system_message = custom_system_message if custom_system_message else default_system_message
            
            # Prepare messages for Converse API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Default inference configuration
            default_inference_config = {
                "maxTokens": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
            
            # Use provided inference config or default
            final_inference_config = inference_config if inference_config else default_inference_config
            
            # Make the Converse Stream API call
            response = self.bedrock_runtime.converse_stream(
                modelId=model_id,
                messages=messages,
                system=[{"text": system_message}],
                inferenceConfig=final_inference_config
            )
            
            # Process streaming response
            for event in response['stream']:
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    if 'text' in delta:
                        yield delta['text']
                elif 'messageStop' in event:
                    break
                    
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            st.error(error_msg)
            yield error_msg
    
    def list_available_models(self) -> list:
        """
        List available Bedrock models
        
        Returns:
            List of available model IDs
        """
        try:
            # Use the same region as the runtime client
            bedrock = boto3.client('bedrock', region_name=self.region_name)
            response = bedrock.list_foundation_models()
            return [model['modelId'] for model in response['modelSummaries']]
        except Exception as e:
            st.error(f"Error listing models: {str(e)}")
            return []