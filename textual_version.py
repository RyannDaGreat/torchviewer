from textual.app import App, ComposeResult
from textual.widgets import Tree, Footer, Input
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen, ModalScreen
import diffusers
from typing import Dict, Optional, Any, Callable
import re

class ModuleNode:
    """Node representing a PyTorch module"""
    def __init__(self, name: str, module_type: str, extra_info: str = ""):
        self.name = name
        self.module_type = module_type
        self.extra_info = extra_info
        self.highlight_color = None

    def get_label(self) -> str:
        """Get the display label for this node"""
        if self.name:
            node_text = f"({self.name}): {self.module_type}"
        else:
            node_text = self.module_type
            
        if self.extra_info:
            if len(self.extra_info) > 30:  # Truncate long extra info
                node_text += f"({self.extra_info[:30]}...)"
            else:
                node_text += f"({self.extra_info})"
                
        return node_text

class ModelTreeViewer(App):
    """Textual app to display a PyTorch model structure as a tree with folding"""
    
    CSS = """
    Tree {
        width: 100%;
        height: 100%;
    }
    
    .node-blue {
        color: blue;
    }
    
    .node-green {
        color: green;
    }
    
    .node-yellow {
        color: yellow;
    }
    
    .node-red {
        color: red;
    }
    
    .node-cyan {
        color: cyan;
    }
    
    .node-magenta {
        color: magenta;
    }
    
    #search-container {
        width: 60%;
        height: auto;
        background: $surface;
        padding: 1 2;
        margin: 1 2;
        box-sizing: border-box;
    }
    
    Input {
        width: 100%;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f", "find", "Find"),
        Binding("c", "cycle_color", "Cycle Color"),
        Binding("n", "find_next", "Next Match"),
    ]
    
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.search_results = []
        self.current_search_idx = -1
        
        # Cache node_data for each tree node
        self.node_data: Dict[Any, ModuleNode] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the UI"""
        yield Tree("Model Structure")
        yield Footer()
    
    def on_mount(self) -> None:
        """Populate the tree when app is mounted"""
        tree = self.query_one(Tree)
        tree.root.expand()
        
        if self.model:
            self.populate_tree(self.model, tree.root)
    
    def populate_tree(self, module, tree_node):
        """Recursively populate tree from PyTorch module"""
        module_name = module._get_name()
        extra_info = module.extra_repr()
        
        # Create or get the parent node data
        if tree_node == self.query_one(Tree).root:
            node_name = ""
        else:
            node_label = str(tree_node.label)
            if ":" in node_label:
                node_name = node_label.split(":")[0].strip("() ")
            else:
                node_name = node_label
        
        node_data = ModuleNode(node_name, module_name, extra_info)
        self.node_data[tree_node] = node_data
        
        # Set node display text
        tree_node.label = node_data.get_label()
        
        # Add all child modules
        for key, child_module in module._modules.items():
            child_node = tree_node.add(f"({key}): {child_module._get_name()}")
            self.populate_tree(child_module, child_node)
    
    def action_find(self) -> None:
        """Prompt for search string"""
        self.push_screen(SearchScreen(self.search_in_tree))
    
    def search_in_tree(self, search_text: str) -> None:
        """Search for nodes matching the given text"""
        if not search_text:
            return
            
        tree = self.query_one(Tree)
        self.search_results = []
        pattern = re.compile(search_text, re.IGNORECASE)
        
        # Helper function to search recursively
        def search_node(node):
            if node in self.node_data:
                node_data = self.node_data[node]
                text = f"{node_data.name} {node_data.module_type} {node_data.extra_info}"
                if pattern.search(text):
                    self.search_results.append(node)
            
            # Search children
            for child in node.children:
                search_node(child)
        
        # Start search from root
        search_node(tree.root)
        
        # Jump to first result
        if self.search_results:
            self.current_search_idx = 0
            self.goto_search_result()
            self.notify(f"Found {len(self.search_results)} matches")
        else:
            self.notify(f"No matches for '{search_text}'")
    
    def action_find_next(self) -> None:
        """Go to next search result"""
        if not self.search_results:
            self.notify("No search results")
            return
            
        self.current_search_idx = (self.current_search_idx + 1) % len(self.search_results)
        self.goto_search_result()
        self.notify(f"Match {self.current_search_idx + 1}/{len(self.search_results)}")
    
    def goto_search_result(self) -> None:
        """Jump to the current search result"""
        if not self.search_results or self.current_search_idx < 0:
            return
            
        node = self.search_results[self.current_search_idx]
        
        # Expand all parent nodes
        parent = node.parent
        while parent:
            parent.expand()
            parent = parent.parent
        
        # Select the node
        tree = self.query_one(Tree)
        tree.select_node(node)
        tree.scroll_to_node(node)
    
    def action_cycle_color(self) -> None:
        """Cycle highlight color of selected node"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        node = tree.cursor_node
        if node not in self.node_data:
            return
            
        node_data = self.node_data[node]
        colors = [None, "blue", "green", "yellow", "red", "cyan", "magenta"]
        
        # Get current color index
        current_idx = 0
        if node_data.highlight_color in colors:
            current_idx = colors.index(node_data.highlight_color)
        
        # Cycle to next color
        next_idx = (current_idx + 1) % len(colors)
        node_data.highlight_color = colors[next_idx]
        
        # Remove all color classes
        for color in colors[1:]:  # Skip None
            if color:
                node.remove_class(f"node-{color}")
        
        # Apply new color class if not None
        if node_data.highlight_color:
            node.add_class(f"node-{node_data.highlight_color}")


class SearchScreen(ModalScreen):
    """Modal screen for searching the tree"""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Search"),
    ]
    
    def __init__(self, on_search: Callable[[str], None]):
        super().__init__()
        self.on_search = on_search
    
    def compose(self) -> ComposeResult:
        yield Container(
            Input(placeholder="Enter search text..."),
            id="search-container"
        )
    
    def on_mount(self) -> None:
        # Focus the input when the screen appears
        self.query_one(Input).focus()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Called when user presses Enter in the input field
        self.on_search(event.value)
        self.dismiss()
    
    def action_cancel(self) -> None:
        # Called when user presses Escape
        self.dismiss()
    
    def action_submit(self) -> None:
        # Called when user presses Enter with the screen focused
        input_value = self.query_one(Input).value
        self.on_search(input_value)
        self.dismiss()


def explore_module(module):
    """Start the interactive module explorer"""
    app = ModelTreeViewer(module)
    app.run()


if __name__ == "__main__":
    # Load the model
    model = diffusers.CogVideoXTransformer3DModel()
    
    # Run the interactive explorer
    explore_module(model)