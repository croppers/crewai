import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
from langchain_ollama import OllamaLLM

# Load environment variables (e.g. SERPER_API_KEY) from a .env file
load_dotenv()
print("SERPER_API_KEY loaded:", os.environ.get("SERPER_API_KEY"))

# Debug wrapper for SerperDevTool (prints and logs raw output)
class DebugSerperDevTool(SerperDevTool):
    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        print("\n[DEBUG] Raw SerperDevTool output:\n", result, "\n")
        with open("serper_raw_output.txt", "w") as f:
            f.write(str(result))
        return result

# Initialize the SerperDevTool (using the API key from .env) and Ollama (gemma:2b) for summarization
search_tool = DebugSerperDevTool(n_results=3)
llm = OllamaLLM(model="gemma:2b", temperature=0.3, top_p=0.9)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "")
    if not query:
        return jsonify({"summary": "Please enter a query.", "search_results": None})
    # Use the SerperDevTool to search (and log raw output) for the query
    raw_search = search_tool.run(search_query=query)
    # (In a production app, you'd parse the raw JSON and extract titles, links, and snippets.)
    # For demo purposes, we'll return the raw (JSON) output as a string.
    search_results = str(raw_search)
    # Use Ollama (gemma:2b) to summarize the search results (or, in a real app, parse and summarize the JSON)
    summary = llm.predict("Summarize the following search results (in a concise, friendly tone) for a user query (" + query + "):\n" + search_results)
    return jsonify({"summary": summary, "search_results": search_results})

if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0") 