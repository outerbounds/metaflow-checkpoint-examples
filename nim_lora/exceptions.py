GATED_HF_ORGS = ["meta-llama"]


class GatedRepoError(Exception):
    """Exception raised for errors related to gated repositories."""

    def __init__(self, repo_id):
        self.message = f"Access to {repo_id} requires the HF_TOKEN environment variable to be set. If you have not done so, you'll need to go to the {repo_id} repo and request access."
        super().__init__(self.message)
