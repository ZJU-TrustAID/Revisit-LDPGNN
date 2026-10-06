import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parent
REPO_LOCAL_FEATURE_REWRITE_ROOT = REPO_ROOT
REPO_LOCAL_FEATURE_REWRITE_SRC = REPO_ROOT / 'src'


@dataclass(frozen=True)
class RepoLocalFeatureRewriteAPI:
    module: ModuleType
    rewrite_features: Callable


_repo_local_feature_rewrite_api = None


def _module_origin_path(module_name, module):
    module_path = getattr(module, '__file__', None)
    if module_path is None:
        raise RuntimeError(f'Imported {module_name} has no __file__; unable to verify import origin.')
    return Path(module_path).resolve()


def _assert_module_under_repo_root(module_name, module):
    module_path = _module_origin_path(module_name, module)
    repo_root = REPO_LOCAL_FEATURE_REWRITE_ROOT.resolve()
    if module_path != repo_root and repo_root not in module_path.parents:
        raise RuntimeError(
            f'Imported {module_name} from unexpected location: {module_path}. '
            f'Expected path under {repo_root}.'
        )


def _purge_modules(prefix):
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(f'{prefix}.'):
            del sys.modules[module_name]


def load_repo_local_feature_rewrite_api():
    global _repo_local_feature_rewrite_api

    if _repo_local_feature_rewrite_api is not None:
        return _repo_local_feature_rewrite_api

    if not REPO_LOCAL_FEATURE_REWRITE_ROOT.is_dir():
        raise RuntimeError(
            f'artificial-node-feature_generator repository root not found at {REPO_LOCAL_FEATURE_REWRITE_ROOT}. '
            'Please initialize/update the repository checkout first.'
        )
    if not REPO_LOCAL_FEATURE_REWRITE_SRC.is_dir():
        raise RuntimeError(
            f'artificial-node-feature-generator source directory not found at {REPO_LOCAL_FEATURE_REWRITE_SRC}. '
            'Expected a repo-local source tree.'
        )

    repo_local_src = str(REPO_LOCAL_FEATURE_REWRITE_SRC)
    if repo_local_src in sys.path:
        sys.path.remove(repo_local_src)
    sys.path.insert(0, repo_local_src)

    _purge_modules('artificial_node_feature_generator')

    try:
        module = importlib.import_module('artificial_node_feature_generator')
    except Exception as exc:
        raise ImportError(
            'Failed to import artificial_node_feature_generator from repo-local source path. '
            f'Expected path: {REPO_LOCAL_FEATURE_REWRITE_SRC}'
        ) from exc

    _assert_module_under_repo_root('artificial_node_feature_generator', module)
    if not hasattr(module, 'rewrite_features'):
        raise RuntimeError('artificial_node_feature_generator is missing expected attribute: rewrite_features')

    _repo_local_feature_rewrite_api = RepoLocalFeatureRewriteAPI(
        module=module,
        rewrite_features=module.rewrite_features,
    )
    return _repo_local_feature_rewrite_api
