from textual.app import App, ComposeResult
from textual.widgets import Tree, Footer
from textual.binding import Binding
import diffusers
from typing import Dict, Optional, Any

class ModuleNode:
    """Node representing a PyTorch module"""
    def __init__(self, name: str, module_type: str, extra_info: str = ""):
        self.name = name
        self.module_type = module_type
        self.extra_info = extra_info

    def get_label(self) -> str:
        """Get the display label for this node"""
        if self.name:
            node_text = f"({self.name}): {self.module_type}"
        else:
            node_text = self.module_type
            
        if self.extra_info:
            # Don't truncate the extra info anymore
            node_text += f"({self.extra_info})"
                
        return node_text

class ModelTreeViewer(App):
    """Textual app to display a PyTorch model structure as a tree with folding"""
    
    CSS = """
    Tree {
        width: 100%;
        height: 100%;
        overflow-x: auto;  /* Allow horizontal scrolling */
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
    




def explore_module(module):
    """Start the interactive module explorer"""
    app = ModelTreeViewer(module)
    app.run()


if __name__ == "__main__":
    # Load the model
    model = diffusers.CogVideoXTransformer3DModel()
    
    # Run the interactive explorer
    explore_module(model)
