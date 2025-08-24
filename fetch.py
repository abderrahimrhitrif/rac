import os
import zipfile
import requests


class GithubFetcher:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.headers = {
            "Accept": "application/vnd.github+json",
        }

    def _get_repo_details(self, repo_full_name: str):
        repo_api_url = f"https://api.github.com/repos/{repo_full_name}"
        print(f"Fetching repo details for '{repo_full_name}'...")
        resp = requests.get(repo_api_url, headers=self.headers)
        if resp.status_code != 200:
            print("Error:", resp.json())
            exit(1)
        return resp.json()

    def _download_repo_zip(self, repo_full_name: str, branch: str):
        repo_name_safe = repo_full_name.replace("/", "_")
        zip_url = f"https://github.com/{repo_full_name}/archive/refs/heads/{branch}.zip"
        zip_path = os.path.join(self.output_dir, f"{repo_name_safe}.zip")
        if os.path.exists(zip_path):
            print(f"Skipping {repo_name_safe} (already downloaded)")
            return zip_path
        print(f"Downloading {repo_name_safe}...")
        zip_resp = requests.get(zip_url, stream=True)
        if zip_resp.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in zip_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {zip_path}")
        else:
            print(f"Failed to download {repo_name_safe}: {zip_resp.status_code}")
            return None
        return zip_path

    def _unzip_and_cleanup(self, zip_path: str):
        extract_path = os.path.join(zip_path[:-4])
        print(f"Unzipping {zip_path} to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Deleting {zip_path}...")
            os.remove(zip_path)
        except zipfile.BadZipFile:
            print(f"{zip_path}: Not a valid zip file.")

    def fetch(self, repo_full_name: str, branch: str = None):
        """
        Fetches a GitHub repository as a ZIP file and saves it locally.
        """
        repo_data = self._get_repo_details(repo_full_name)
        default_branch = branch if branch else repo_data["default_branch"]
        zip_path = self._download_repo_zip(repo_full_name, default_branch)
        if zip_path:
            self._unzip_and_cleanup(zip_path)
