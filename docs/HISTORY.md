# History provenance

The available artifacts are **source-code ZIP snapshots**, not a `.git` repository. This packaged repository contains Git commits recording the reconstructed milestones:

1. Initial synthetic intake LoRA source snapshot.
2. External ConstructCIE/OSHA source snapshot.
3. v2.1 evaluation-hardening source snapshot with the audit hotfix.
4. Shareable consolidation and documentation.

These commits are a **reconstructed archive timeline**, not the author's original commit history. Their author identity and timestamps must not be represented as historical contributions. Genuine local Git history lives only in the owner's original `.git` database or a remote repository. To preserve that history, apply the new tree to the original working checkout using `scripts/apply-to-existing-checkout.sh` and commit there; never replace the original `.git` directory with the one from this snapshot.
