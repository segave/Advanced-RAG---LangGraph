from dotenv import load_dotenv

load_dotenv()

from backend.graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    #print(app.invoke(input={"question": "what is agent memory?"}))
    print(app.invoke(input={"question": "what is a good way to make pizza?"}))
