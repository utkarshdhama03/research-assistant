"""Exceptions raised by the data layer."""


class RepositoryError(Exception):
    """Base class for data-layer errors."""


class NotFoundError(RepositoryError, LookupError):
    """A referenced row does not exist."""


class EvidenceError(RepositoryError):
    """Claim evidence violates the citation invariant."""


class QuoteMismatchError(EvidenceError):
    """The quote is empty, its offsets are out of range, or it is not chunk.text[start:end]."""


class ChunkNotInRunError(EvidenceError):
    """The cited chunk belongs to a different run than the claim."""


class SubQuestionNotInRunError(RepositoryError):
    """The sub-question belongs to a different run than the claim."""
