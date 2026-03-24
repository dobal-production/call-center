import streamlit as st
import os
import yaml
from utils.bedrock import BedrockClient

def load_transcript_file(file_path: str) -> str:
    """
    Load transcript text from file
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        Transcript text content
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return "녹취 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"파일 읽기 오류: {str(e)}"

def load_prompt_examples(file_path: str = "prompt_examples.yaml") -> dict:
    """
    Load prompt examples from YAML file
    
    Args:
        file_path: Path to the YAML file containing prompt examples
        
    Returns:
        Dictionary containing prompt examples
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            return {"prompt_examples": {}}
    except Exception as e:
        st.error(f"프롬프트 예제 파일 읽기 오류: {str(e)}")
        return {"prompt_examples": {}}

def main():
    st.set_page_config(
        page_title="고객센터 데모",
        page_icon="📞",
        layout="wide"
    )
    
    st.title("📞 자동차 보험 상담")
    st.markdown("AI-powered call center support system using Amazon Bedrock")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model selection with display names and IDs
        model_options = {
            "Claude Sonnet 4.6": "global.anthropic.claude-sonnet-4-6",
            "Claude Haiku 4.5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Opus 4.6": "global.anthropic.claude-opus-4-6-v1",
            "Nova Pro": "apac.amazon.nova-pro-v1:0",
            "Nova 2 Lite": "global.amazon.nova-2-lite-v1:0"
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            help="Choose the AI model for generating responses"
        )
        
        # Get the actual model ID
        model_id = model_options[selected_model_name]
        
        st.divider()
        
        # Inference parameters
        st.subheader("🎛️ Inference Parameters")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls randomness: lower = more focused, higher = more creative"
        )
        
        st.divider()
        
        # System message customization
        st.subheader("📝 System Prompt")
        custom_system_message = st.text_area(
            "Custom System Prompt (optional)",
            placeholder="Enter custom instructions for the AI assistant...",
            height=100,
            help="Override the default system prompt with custom instructions"
        )
    
    with st.expander("아키텍처 보기"):
        st.image("images/call-center-01.png")

    st.audio("media/3_Inquiries_related_to_premium_surcharge_calculation.mp3", format='audio/mp3')

    # 녹취 파일 경로 설정 (필요에 따라 수정)
    transcript_file_path = "media/3_Inquiries_related_to_premium_surcharge_calculation.txt"
    
    # 녹취 텍스트 로드 (전역적으로 사용)
    transcript_text = load_transcript_file(transcript_file_path)
    
    with st.expander("녹취록 보기"):
        # 녹취 내용 표시
        st.write(transcript_text)

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("프롬프트")
        # 녹취 내용이 복사되었는지 확인하고 기본값으로 설정
        default_text = ""
        if hasattr(st.session_state, 'transcript_copied'):
            default_text = st.session_state.transcript_copied
            # 한 번 사용 후 삭제
            del st.session_state.transcript_copied
            
        user_input = st.text_area(
            "AI에게 지시를 내려주세요.:",
            value=default_text,
            height=250,
            placeholder="Type the customer's inquiry here..."
        )
        
        if st.button("실행", type="primary"):
            if user_input:
                try:
                    bedrock_client = BedrockClient()
                    
                    # Prepare inference config
                    inference_config = {
                        "maxTokens": max_tokens,
                        "temperature": temperature
                    }
                    
                    # Prepare context-aware prompt with transcript
                    context_prompt = f"""다음은 고객과 상담원 간의 통화 녹취록입니다:

<context>
{transcript_text}
</context>

위 <context>내용을 참고하여 다음 요청에 답변해주세요:

{user_input}"""
                    
                    # Always use streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in bedrock_client.generate_response_stream(
                        prompt=context_prompt, 
                        model_id=model_id,
                        inference_config=inference_config,
                        custom_system_message=custom_system_message if custom_system_message else None
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + " ▋")
                    
                    response_placeholder.markdown(full_response)
                    st.success("응답 생성이 완료되었습니다!")
                            
                except Exception as e:
                    st.error(f"오류: {str(e)}")
            else:
                st.warning("프롬프트를 입력해주세요.")
    
    # 프롬프트 예제를 위한 영역
    with col2:
        st.subheader("프롬프트 예제들")
        
        # YAML 파일에서 프롬프트 예제들 로드
        prompt_data = load_prompt_examples()
        prompt_examples = prompt_data.get("prompt_examples", {})
        
        # 각 프롬프트 예제를 expander와 st.code로 표시
        for key, example in prompt_examples.items():
            title = example.get("title", key)
            content = example.get("content", "")
            
            with st.expander(title):
                st.code(content, language="text")
        

if __name__ == "__main__":
    main()