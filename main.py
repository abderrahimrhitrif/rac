import argparse
from rac import RAC

def main():
    parser = argparse.ArgumentParser(description="Code Retrieval and Question Answering")
    parser.add_argument("repo_url", help="URL of the GitHub repository")
    parser.add_argument("question", help="Question to ask about the repository")
    args = parser.parse_args()

    rag = RAC(args.repo_url)
    rag.setup()
    answer = rag.ask(args.question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
