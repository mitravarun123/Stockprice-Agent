"""
Flask GUI for CrewAI Stock Analyzer
=====================================
Run:
  pip install flask
  python app.py
  Open http://localhost:5000
"""

import threading
import queue
import json
from flask import Flask, render_template, request, Response, stream_with_context
from stock_crew import stock_crew

app = Flask(__name__)


# Queue to stream agent progress to the browser
progress_queue = queue.Queue()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    user_input = request.json.get("query", "").strip()
    if not user_input:
        return {"error": "Please enter a stock query"}, 400

    # Run crew in background thread
    result_holder = {}

    def run_crew():
        try:
            result = stock_crew.kickoff(inputs={"user_input": user_input})
            result_holder["output"] = str(result)
        except Exception as e:
            result_holder["output"] = f"Error: {str(e)}"

    thread = threading.Thread(target=run_crew)
    thread.start()
    thread.join()

    return {"result": result_holder.get("output", "No result")}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
