from textual.app import App, ComposeResult
from textual.widgets import Tree, Footer
from textual.binding import Binding
import diffusers
from typing import Dict, Optional, Any
import torch

class ModuleNode:
    """Node representing a PyTorch module"""
    def __init__(self, name: str, module_type: str, extra_info: str = "", state_dict_info: str = ""):
        self.name = name
        self.module_type = module_type
        self.extra_info = extra_info
        self.state_dict_info = state_dict_info

    def get_label(self, show_tensor_shapes=True) -> str:
        """Get the display label for this node"""
        if self.name:
            node_text = f"({self.name}): {self.module_type}"
        else:
            node_text = self.module_type
            
        if self.extra_info:
            # Don't truncate the extra info anymore
            node_text += f"({self.extra_info})"
                
        if show_tensor_shapes and self.state_dict_info:
            node_text += f" | [tensor-shape]{self.state_dict_info}[/tensor-shape]"
                
        return node_text

class ModelTreeViewer(App):
    """Textual app to display a PyTorch model structure as a tree with folding"""
    
    CSS = """
    Tree {
        width: 100%;
        height: 100%;
        overflow-x: auto;  /* Allow horizontal scrolling */
    }
    
    .tensor-shape {
        color: #5fd700;
        text-style: italic;
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
        Binding("T", "toggle_same_class", "Toggle Same Class Type"),
        
        # Display options
        Binding("t", "toggle_tensor_shapes", "Toggle Tensor Shapes"),
    ]
    
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        
        # Cache node_data for each tree node
        self.node_data: Dict[Any, ModuleNode] = {}
        
        # Flag to control whether to show tensor shapes
        self.show_tensor_shapes = True
    
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
        
        # Get state dict info for leaf modules (no children)
        state_dict_info = ""
        if not module._modules:  # This is a leaf module
            try:
                # Get state dict and format tensor shapes
                params = list(module.state_dict().items())
                if params:
                    shapes = []
                    for param_name, tensor in params:
                        if isinstance(tensor, torch.Tensor):
                            shape_str = f"{param_name}: {tuple(tensor.shape)}"
                            shapes.append(shape_str)
                    state_dict_info = ", ".join(shapes)
            except Exception as e:
                state_dict_info = f"Error: {str(e)}"
        
        # Create or get the parent node data
        if tree_node == self.query_one(Tree).root:
            node_name = ""
        else:
            node_label = str(tree_node.label)
            if ":" in node_label:
                node_name = node_label.split(":")[0].strip("() ")
            else:
                node_name = node_label
        
        node_data = ModuleNode(node_name, module_name, extra_info, state_dict_info)
        self.node_data[tree_node] = node_data
        
        # Set node display text
        tree_node.label = node_data.get_label(show_tensor_shapes=self.show_tensor_shapes)
        
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
    
    def toggle_node_state(self, node, expand: Optional[bool] = None) -> bool:
        """Toggle or set a node's expand/collapse state
        
        Args:
            node: The tree node to toggle
            expand: If True, expand; if False, collapse; if None, toggle current state
            
        Returns:
            True if action was performed, False otherwise
        """
        if not node:
            return False
            
        if expand is None:
            # Toggle current state
            if node.is_expanded:
                node.collapse()
                return True
            elif node.allow_expand:
                node.expand()
                return True
        elif expand and not node.is_expanded and node.allow_expand:
            node.expand()
            return True
        elif not expand and node.is_expanded:
            node.collapse()
            return True
            
        return False
    
    def action_toggle_node(self) -> None:
        """Toggle expand/collapse of selected node"""
        tree = self.query_one(Tree)
        self.toggle_node_state(tree.cursor_node)
    
    def process_nodes_recursively(self, start_node, expand: Optional[bool], subtree_only: bool = False) -> int:
        """Process nodes recursively to expand or collapse them
        
        Args:
            start_node: The node to start processing from
            expand: If True, expand nodes; if False, collapse nodes
            subtree_only: If True, only process the subtree under start_node
            
        Returns:
            Number of nodes processed
        """
        count = 0
        
        def process_node(node):
            nonlocal count
            if self.toggle_node_state(node, expand):
                count += 1
            for child in node.children:
                process_node(child)
                
        # Start processing from specified node
        process_node(start_node)
        return count
    
    def action_collapse_all(self) -> None:
        """Collapse all nodes"""
        tree = self.query_one(Tree)
        self.process_nodes_recursively(tree.root, expand=False)
        self.notify("All nodes collapsed")
    
    def action_expand_all(self) -> None:
        """Expand all nodes"""
        tree = self.query_one(Tree)
        
        # We need to do this in multiple passes to ensure all nodes get expanded
        # Because expanding a node may create new children
        for _ in range(10):  # Limit to 10 passes to avoid infinite loop
            if self.process_nodes_recursively(tree.root, expand=True) == 0:
                break
                
        self.notify("All nodes expanded")
    
    def action_collapse_subtree(self) -> None:
        """Collapse all nodes in the current subtree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        count = self.process_nodes_recursively(tree.cursor_node, expand=False)
        self.notify(f"Collapsed {count} nodes under {tree.cursor_node.label}")
    
    def action_expand_subtree(self) -> None:
        """Expand all nodes in the current subtree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return
            
        # We may need multiple passes to fully expand all levels
        for _ in range(10):  # Limit to 10 passes to avoid infinite loop
            if self.process_nodes_recursively(tree.cursor_node, expand=True) == 0:
                break
                
        self.notify(f"Expanded subtree under {tree.cursor_node.label}")
    
    def process_nodes_by_type(self, start_node, target_type: str, expand: bool) -> int:
        """Process all nodes of a specific type in a subtree
        
        Args:
            start_node: Node to start processing from
            target_type: Module type to match
            expand: Whether to expand (True) or collapse (False)
            
        Returns:
            Number of nodes processed
        """
        count = 0
        
        def process_node(node):
            nonlocal count
            # Check if this node matches the target type
            if node in self.node_data and self.node_data[node].module_type == target_type:
                if self.toggle_node_state(node, expand):
                    count += 1
            
            # Process all children
            for child in node.children:
                process_node(child)
        
        # Start from specified node
        process_node(start_node)
        return count
    
    def action_toggle_same_class(self) -> None:
        """Toggle all nodes with the same class type in the entire tree"""
        tree = self.query_one(Tree)
        if not tree.cursor_node or tree.cursor_node not in self.node_data:
            return
            
        current_node = tree.cursor_node
        target_module_type = self.node_data[current_node].module_type
        
        # Toggle based on current node's state
        expand = not current_node.is_expanded
        
        # Process nodes of the same type in the entire tree
        count = self.process_nodes_by_type(tree.root, target_module_type, expand)
        
        action = "Expanded" if expand else "Collapsed"
        self.notify(f"{action} {count} nodes of class {target_module_type}")
    
    def action_toggle_tensor_shapes(self) -> None:
        """Toggle display of tensor shapes in the tree"""
        self.show_tensor_shapes = not self.show_tensor_shapes
        
        # Update all node labels
        tree = self.query_one(Tree)
        
        def update_node_label(node):
            if node in self.node_data:
                node.label = self.node_data[node].get_label(show_tensor_shapes=self.show_tensor_shapes)
            for child in node.children:
                update_node_label(child)
        
        update_node_label(tree.root)
        
        status = "Showing" if self.show_tensor_shapes else "Hiding"
        self.notify(f"{status} tensor shapes")



def explore_module(module):
    """Start the interactive module explorer"""
    app = ModelTreeViewer(module)
    app.run()


if __name__ == "__main__":
    # Load the model
    model = diffusers.CogVideoXTransformer3DModel()
    
    # Run the interactive explorer
    explore_module(model)
