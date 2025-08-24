import argparse
from rac import RAC
import warnings

def main():
    # suppress warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Code Retrieval and Question Answering")
    parser.add_argument("repo_url", help="URL of the GitHub repository")
    parser.add_argument("model_name", help="name of the model to use", default="Qwen3:0.6B")
    parser.add_argument("question", help="Question to ask about the repository")
    args = parser.parse_args()

    rag = RAC(repo_name=args.repo_url, model=args.model_name)
    rag.setup()
    answer = rag.ask(args.question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
