import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
            /* Base */
            .stApp {
                background-color: #F8F9FA;
                color: #212529;
                font-family: 'Inter', -apple-system, sans-serif;
            }
            
            /* Contenedores */
            .stContainer {
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            /* Inputs */
            .stTextInput > div > div > input {
                background-color: #FFFFFF;
                color: #212529;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                padding: 0.75rem;
                font-size: 1rem;
                transition: all 0.2s ease;
            }
            .stTextInput > div > div > input:focus {
                border-color: #3B82F6;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
            }
            
            /* Botones */
            .stButton > button {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.2s ease;
                width: 100%;
            }
            .stButton > button:hover {
                background-color: #2563EB;
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
            }
            
            /* Chat */
            .stMessage {
                background-color: #FFFFFF !important;
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #E5E7EB;
            }
            
            /* Usuario vs AI */
            div[data-testid="stChatMessage"] {
                background-color: #F3F4F6;
                padding: 1rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                border: 1px solid #E5E7EB;
            }
            div[data-testid="stChatMessage"].user {
                background-color: #EFF6FF;
                border-color: #BFDBFE;
            }
            
            /* Sidebar */
            .stSidebar {
                background-color: #FFFFFF;
                border-right: 1px solid #E5E7EB;
                padding: 2rem 1rem;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: #1F2937;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            
            /* Links */
            a {
                color: #3B82F6;
                text-decoration: none;
                transition: color 0.2s ease;
            }
            a:hover {
                color: #2563EB;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #F1F1F1;
            }
            ::-webkit-scrollbar-thumb {
                background: #C5C5C5;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #A8A8A8;
            }

            /* Divisores */
            hr {
                border-color: #E5E7EB;
                margin: 1.5rem 0;
            }

            /* Tooltips y elementos flotantes */
            div[data-baseweb="tooltip"] {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            }

            /* Text Areas */
            .stTextArea > div > div > textarea {
                background-color: #f0f2f6;
            }

            /* Sidebar Inputs */
            .sidebar .stTextInput > div > div > input {
                background-color: #f0f2f6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
