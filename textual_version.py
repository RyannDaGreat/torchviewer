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
        # Quit
        Binding("q", "quit", "Quit"),
        
        # Navigation - vim style
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
        Binding("h", "cursor_left", "Left"),
        Binding("l", "cursor_right", "Right"),
        
        # Search
        Binding("/", "find", "Find"),
        Binding("n", "find_next", "Next Match"),
        Binding("p", "find_prev", "Prev Match"),
        
        # Coloring
        Binding("c", "cycle_color", "Cycle Color"),
        
        # Expand/Collapse
        Binding("space", "toggle_node", "Toggle Node"),
        Binding("z", "collapse_all", "Collapse All"),
        Binding("Z", "expand_all", "Expand All"),
        Binding("s", "collapse_subtree", "Collapse Subtree"),
        Binding("S", "expand_subtree", "Expand Subtree"),
        Binding("t", "toggle_same_name_subtree", "Toggle Same Name in Subtree"),
        Binding("T", "toggle_same_name_all", "Toggle Same Name Everywhere"),
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
        has_children = bool(module._modules)
        
        # Set whether this node can be expanded
        tree_node.allow_expand = has_children
        
        # Add all child modules
        for key, child_module in module._modules.items():
            child_node = tree_node.add(f"({key}): {child_module._get_name()}")
            self.populate_tree(child_module, child_node)
    
    def action_find(self) -> None:
        """Prompt for search string"""
        try:
            self.push_screen(SearchScreen(self.search_in_tree))
        except Exception as e:
            self.notify(f"Search error: {str(e)}", severity="error")
    
    def search_in_tree(self, search_text: str) -> None:
        """Search for nodes matching the given text"""
        if not search_text:
            return
        
        try:
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
        except Exception as e:
            self.notify(f"Search error: {str(e)}", severity="error")
    
    def action_find_next(self) -> None:
        """Go to next search result"""
        if not self.search_results:
            self.notify("No search results")
            return
            
        self.current_search_idx = (self.current_search_idx + 1) % len(self.search_results)
        self.goto_search_result()
        self.notify(f"Match {self.current_search_idx + 1}/{len(self.search_results)}")
        
    def action_find_prev(self) -> None:
        """Go to previous search result"""
        if not self.search_results:
            self.notify("No search results")
            return
            
        self.current_search_idx = (self.current_search_idx - 1) % len(self.search_results)
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
    
    def action_cursor_down(self) -> None:
        """Move cursor down (j key)"""
        tree = self.query_one(Tree)
        tree.action_cursor_down()
    
    def action_cursor_up(self) -> None:
        """Move cursor up (k key)"""
        tree = self.query_one(Tree)
        tree.action_cursor_up()
    
    def action_cursor_left(self) -> None:
        """Move cursor left/collapse (h key)"""
        tree = self.query_one(Tree)
        node = tree.cursor_node
        if node and node.is_expanded:
            node.collapse()  # Use node.collapse() directly
        elif node and node.parent:
            tree.select_node(node.parent)
            tree.scroll_to_node(node.parent)
    
    def action_cursor_right(self) -> None:
        """Move cursor right/expand (l key)"""
        tree = self.query_one(Tree)
        node = tree.cursor_node
        if node and not node.is_expanded and node.allow_expand:
            node.expand()  # Use node.expand() directly
    
    def action_toggle_node(self) -> None:
        """Toggle expand/collapse of selected node"""
        tree = self.query_one(Tree)
        node = tree.cursor_node
        if node:
            if node.is_expanded:
                node.collapse()  # Use node.collapse() directly
            elif node.allow_expand:
                node.expand()  # Use node.expand() directly
    
    def action_collapse_all(self) -> None:
        """Collapse all nodes"""
        tree = self.query_one(Tree)
        
        # We can't use walk_tree - use a recursive function instead
        def collapse_all_nodes(node):
            if hasattr(node, "is_expanded") and node.is_expanded:
                node.collapse()  # Use node.collapse() directly
            for child in node.children:
                collapse_all_nodes(child)
        
        # Start from the root
        collapse_all_nodes(tree.root)
        self.notify("All nodes collapsed")
    
    def action_expand_all(self) -> None:
        """Expand all nodes"""
        tree = self.query_one(Tree)
        
        # We need to do this in multiple passes to ensure all nodes get expanded
        # Because expanding a node may create new children
        def expand_pass():
            # Keep track of whether we expanded anything this pass
            expanded = False
            
            def expand_nodes(node):
                nonlocal expanded
                if hasattr(node, "is_expanded") and not node.is_expanded and node.allow_expand:
                    node.expand()  # Use node.expand() directly
                    expanded = True
                for child in node.children:
                    expand_nodes(child)
                    
            # Start from the root
            expand_nodes(tree.root)
            return expanded
        
        # Run multiple passes until no more expansions occur
        for _ in range(10):  # Limit to 10 passes to avoid infinite loop
            if not expand_pass():
                break
                
        self.notify("All nodes expanded")
    
    def action_collapse_subtree(self) -> None:
        """Collapse all nodes in the current subtree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        # Start from the current node
        current_node = tree.cursor_node
        
        # Function to recursively collapse a node and all its children
        def collapse_recursive(node):
            if hasattr(node, "is_expanded") and node.is_expanded:
                node.collapse()  # Use node.collapse() directly
            
            # Now collapse all children
            for child in node.children:
                collapse_recursive(child)
        
        # Start collapsing from the current node
        collapse_recursive(current_node)
                
        self.notify(f"Collapsed subtree under {current_node.label}")
    
    def action_expand_subtree(self) -> None:
        """Expand all nodes in the current subtree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        # Start from the current node
        current_node = tree.cursor_node
        
        # Function to recursively expand a node and all its children
        def expand_recursive(node):
            if hasattr(node, "is_expanded") and not node.is_expanded and node.allow_expand:
                node.expand()  # Use node.expand() directly
            
            # Now expand all children
            for child in node.children:
                expand_recursive(child)
        
        # Start expanding from the current node
        expand_recursive(current_node)
        
        self.notify(f"Expanded subtree under {current_node.label}")
    
    def action_toggle_same_name_subtree(self) -> None:
        """Toggle all nodes with the same name in the current subtree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        current_node = tree.cursor_node
        if current_node not in self.node_data:
            return
            
        # Get the node type we're looking for
        target_module_type = self.node_data[current_node].module_type
        
        # Get current expand state to toggle
        expand = not current_node.is_expanded
        
        # Find all nodes in the subtree with the same module type
        count = 0
        
        def process_subtree(node):
            nonlocal count
            # Check if this node matches
            if node in self.node_data and self.node_data[node].module_type == target_module_type:
                if expand and hasattr(node, "is_expanded") and not node.is_expanded and node.allow_expand:
                    node.expand()  # Use node.expand() directly
                    count += 1
                elif not expand and hasattr(node, "is_expanded") and node.is_expanded:
                    node.collapse()  # Use node.collapse() directly
                    count += 1
            
            # Process all children
            for child in node.children:
                process_subtree(child)
        
        # Start from current node
        process_subtree(current_node)
        
        action = "Expanded" if expand else "Collapsed"
        self.notify(f"{action} {count} nodes of type {target_module_type} in subtree")
    
    def action_toggle_same_name_all(self) -> None:
        """Toggle all nodes with the same name in the entire tree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        current_node = tree.cursor_node
        if current_node not in self.node_data:
            return
            
        # Get the node type we're looking for
        target_module_type = self.node_data[current_node].module_type
        
        # Get current expand state to toggle
        expand = not current_node.is_expanded
        
        # Find all nodes in the entire tree with the same module type
        count = 0
        
        def process_tree(node):
            nonlocal count
            # Check if this node matches
            if node in self.node_data and self.node_data[node].module_type == target_module_type:
                if expand and hasattr(node, "is_expanded") and not node.is_expanded and node.allow_expand:
                    node.expand()  # Use node.expand() directly
                    count += 1
                elif not expand and hasattr(node, "is_expanded") and node.is_expanded:
                    node.collapse()  # Use node.collapse() directly
                    count += 1
            
            # Process all children
            for child in node.children:
                process_tree(child)
        
        # Start from root
        process_tree(tree.root)
        
        action = "Expanded" if expand else "Collapsed"
        self.notify(f"{action} {count} nodes of type {target_module_type} in entire tree")
    
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
        
        # Remove all color classes - use set_classes instead
        node_classes = set()
        if node_data.highlight_color:
            node_classes.add(f"node-{node_data.highlight_color}")
            
        # Apply the classes
        node.set_classes(node_classes)


class SearchScreen(Screen):
    """Screen for searching the tree"""
    
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
        self.dismiss()  # Close the screen
    
    def action_cancel(self) -> None:
        # Called when user presses Escape
        self.dismiss()  # Close the screen
    
    def action_submit(self) -> None:
        # Called when user presses Enter with the screen focused
        input_value = self.query_one(Input).value
        self.on_search(input_value)
        self.dismiss()  # Close the screen


def explore_module(module):
    """Start the interactive module explorer"""
    app = ModelTreeViewer(module)
    app.run()


if __name__ == "__main__":
    # Load the model
    model = diffusers.CogVideoXTransformer3DModel()
    
    # Run the interactive explorer
    explore_module(model)