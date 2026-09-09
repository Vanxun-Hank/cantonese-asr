import pytest
from scripts.train_p2_full import initialize_checkpoint_paths


class Broadcast:
    def __init__(self):
        self.saved = None

    def root(self, status, src):
        self.saved = status.copy()

    def peer(self, status, src):
        status[:] = self.saved


def test_peer_accepts_staging_created_by_rank_zero(tmp_path):
    channel = Broadcast()
    checkpoint, staging = tmp_path / "checkpoint", tmp_path / "staging"
    initialize_checkpoint_paths(checkpoint, staging, 0, 4, channel.root)
    assert staging.is_dir()
    for rank in (1, 2, 3):
        initialize_checkpoint_paths(checkpoint, staging, rank, 4, channel.peer)


def test_existing_evidence_is_preserved_and_failure_broadcast(tmp_path):
    channel = Broadcast()
    checkpoint, staging = tmp_path / "checkpoint", tmp_path / "staging"
    staging.mkdir()
    evidence = staging / "evidence"
    evidence.write_text("preserve")
    for rank, broadcast in ((0, channel.root), (1, channel.peer)):
        with pytest.raises(RuntimeError, match="FileExistsError"):
            initialize_checkpoint_paths(checkpoint, staging, rank, 2, broadcast)
    assert evidence.read_text() == "preserve"
