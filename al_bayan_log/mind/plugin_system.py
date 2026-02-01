"""
Bayan Plugin System
نظام الإضافات للغة بيان

A flexible plugin architecture that allows extending Bayan with custom
functionality, new language constructs, and domain-specific features.
"""

import os
import sys
import importlib
import importlib.util
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported."""
    LANGUAGE_EXTENSION = "language"      # New syntax/keywords
    LIBRARY = "library"                  # New functions/modules
    DOMAIN = "domain"                    # Domain-specific features
    INTEGRATION = "integration"          # External service integration
    VISUALIZATION = "visualization"      # Output visualization
    ANALYSIS = "analysis"                # Code analysis tools


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    license: str = "MIT"
    bayan_version: str = ">=1.0.0"


class PluginInterface(ABC):
    """
    Base interface for all Bayan plugins.
    الواجهة الأساسية لجميع إضافات بيان.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.
        
        Args:
            context: Dictionary containing runtime context (interpreter, etc.)
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def get_functions(self) -> Dict[str, Callable]:
        """
        Return functions to register in Bayan's global scope.
        
        Returns:
            Dictionary mapping function names to callables
        """
        return {}
    
    def get_keywords(self) -> Dict[str, Any]:
        """
        Return new keywords/syntax extensions.
        
        Returns:
            Dictionary of keyword definitions
        """
        return {}
    
    def get_commands(self) -> Dict[str, Callable]:
        """
        Return CLI commands the plugin provides.
        
        Returns:
            Dictionary mapping command names to handlers
        """
        return {}


class PluginRegistry:
    """
    Central registry for managing plugins.
    السجل المركزي لإدارة الإضافات.
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._loaded_keywords: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._plugin_paths: List[Path] = []
    
    def add_plugin_path(self, path: Union[str, Path]) -> None:
        """Add a directory to search for plugins."""
        path = Path(path)
        if path.is_dir() and path not in self._plugin_paths:
            self._plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin paths.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for path in self._plugin_paths:
            for item in path.iterdir():
                if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                    plugin_name = item.stem
                    discovered.append(plugin_name)
                elif item.is_dir() and (item / '__init__.py').exists():
                    plugin_name = item.name
                    discovered.append(plugin_name)
        
        logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def load_plugin(self, name: str, path: Optional[Path] = None) -> bool:
        """
        Load a plugin by name.
        
        Args:
            name: Plugin name
            path: Optional explicit path to plugin
        
        Returns:
            True if loaded successfully
        """
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already loaded")
            return True
        
        try:
            # Find plugin file
            plugin_path = None
            if path:
                plugin_path = path
            else:
                for search_path in self._plugin_paths:
                    candidate = search_path / f"{name}.py"
                    if candidate.exists():
                        plugin_path = candidate
                        break
                    candidate = search_path / name / "__init__.py"
                    if candidate.exists():
                        plugin_path = candidate
                        break
            
            if not plugin_path:
                logger.error(f"Plugin '{name}' not found")
                return False
            
            # Load module
            spec = importlib.util.spec_from_file_location(name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (inspect.isclass(attr) and 
                    issubclass(attr, PluginInterface) and 
                    attr is not PluginInterface):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                logger.error(f"No PluginInterface implementation found in '{name}'")
                return False
            
            # Instantiate and initialize
            plugin = plugin_class()
            if plugin.initialize({}):
                self._plugins[name] = plugin
                self._register_plugin_features(plugin)
                logger.info(f"✅ Loaded plugin: {name} v{plugin.metadata.version}")
                return True
            else:
                logger.error(f"Plugin '{name}' initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load plugin '{name}': {e}")
            return False
    
    def _register_plugin_features(self, plugin: PluginInterface) -> None:
        """Register all features from a plugin."""
        # Register functions
        for func_name, func in plugin.get_functions().items():
            self._loaded_functions[func_name] = func
        
        # Register keywords
        for kw_name, kw_def in plugin.get_keywords().items():
            self._loaded_keywords[kw_name] = kw_def
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self._plugins:
            return False
        
        try:
            plugin = self._plugins[name]
            plugin.shutdown()
            
            # Remove registered features
            for func_name in list(plugin.get_functions().keys()):
                self._loaded_functions.pop(func_name, None)
            
            for kw_name in list(plugin.get_keywords().keys()):
                self._loaded_keywords.pop(kw_name, None)
            
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin '{name}': {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with their metadata."""
        return [
            {
                "name": name,
                "version": plugin.metadata.version,
                "type": plugin.metadata.plugin_type.value,
                "description": plugin.metadata.description,
                "author": plugin.metadata.author
            }
            for name, plugin in self._plugins.items()
        ]
    
    def get_functions(self) -> Dict[str, Callable]:
        """Get all functions from loaded plugins."""
        return self._loaded_functions.copy()
    
    def get_keywords(self) -> Dict[str, Any]:
        """Get all keywords from loaded plugins."""
        return self._loaded_keywords.copy()
    
    # Hook system
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for an event hook."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger an event hook and return results from all callbacks."""
        results = []
        for callback in self._hooks.get(event, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback error for '{event}': {e}")
        return results


# Global plugin registry
_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


def load_plugin(name: str, path: Optional[str] = None) -> bool:
    """Load a plugin by name."""
    return _registry.load_plugin(name, Path(path) if path else None)


def unload_plugin(name: str) -> bool:
    """Unload a plugin."""
    return _registry.unload_plugin(name)


def list_plugins() -> List[Dict[str, Any]]:
    """List all loaded plugins."""
    return _registry.list_plugins()


# ========================
# Example Plugin Template
# ========================

class ExamplePlugin(PluginInterface):
    """
    Example plugin demonstrating the plugin interface.
    مثال على إضافة توضح واجهة الإضافات.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            author="Bayan Team",
            description="An example plugin demonstrating the plugin system",
            plugin_type=PluginType.LIBRARY,
            keywords=["example", "demo"]
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        logger.info("Example plugin initialized")
        return True
    
    def shutdown(self) -> None:
        logger.info("Example plugin shutdown")
    
    def get_functions(self) -> Dict[str, Callable]:
        def greet(name: str) -> str:
            return f"Hello, {name}! From Example Plugin"
        
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        return {
            "plugin_greet": greet,
            "plugin_add": add_numbers
        }


# Initialize default plugin path
def setup_default_plugin_path():
    """Setup the default plugin search path."""
    # User plugins directory
    home = Path.home()
    user_plugins = home / ".bayan" / "plugins"
    if user_plugins.is_dir():
        _registry.add_plugin_path(user_plugins)
    
    # Project plugins directory
    project_plugins = Path(__file__).parent / "plugins"
    if project_plugins.is_dir():
        _registry.add_plugin_path(project_plugins)


# Auto-setup on import
try:
    setup_default_plugin_path()
except:
    pass
