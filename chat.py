import streamlit as st
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import torch
st.title("Model")

# 初始化所有必要的会话状态变量
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
    
if 'model' not in st.session_state:
    st.session_state.model = None

# 只加载一次模型
if st.session_state.tokenizer is None or st.session_state.model is None :
    with st.spinner("正在加载模型，请稍候..."):
        try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "/path/to/model", 
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    "/path/to/model",
                    trust_remote_code=True,
                    torch_dtype="auto"
                ).eval()
                
                # 将模型和分词器保存到会话状态
                if torch.cuda.is_available():
                    st.session_state.tokenizer = tokenizer.cuda()
                    st.session_state.model = model.cuda()
                else:
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                st.success(f"模型加载成功！")
            
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.stop()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("输入您的问题..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
    
    # 准备模型回复
    with st.chat_message("Assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 使用分词器处理输入
            inputs = st.session_state.tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(st.session_state.model.device)
            
            # 生成回复
            generated_ids = st.session_state.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                top_p=0.7,
                temperature=0.95,
                repetition_penalty=1.1
            )
            
            # 解码生成的文本
            full_response = st.session_state.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # 显示回复（模拟流式效果）
            for i in range(0, len(full_response), 5):
                message_placeholder.markdown(full_response[:i] + "▌")
                st.session_state.model.eval()  # 确保模型处于评估模式
            
            # 最终显示完整回复
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"生成回复时出错: {str(e)}")
            full_response = "抱歉，生成回复时出现问题"
            message_placeholder.markdown(full_response)
    
    # 添加AI回复到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})