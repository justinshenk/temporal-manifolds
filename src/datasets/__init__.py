"""Dataset generation: parametric prompt datasets and default experiment configs.

Parametric generation (generate.py / phrasings.py / templates.py): a *scenario*
defines a prompt template parameterised by a planning horizon (in months). For
every horizon we emit one prompt per *phrasing* in the equivalent-phrasing
group ("4 weeks" ≡ "1 month" ≡ "28 days"). Train/test split is by
phrasing-group, so held-out phrasings exercise whether equivalent phrasings
collapse onto the same point on the manifold.

Default configs (default_configs.py / default_datasets.py): canonical dataset
and experiment configurations ported from temporal-awareness.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from ..common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
