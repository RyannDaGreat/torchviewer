import curses
import diffusers
import re
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Terminal ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
}

def fansi(text: str, color: str) -> str:
    """Format text with ANSI color codes"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def indentify(text: str, indent: str) -> str:
    """Add indentation to each line of text"""
    return '\n'.join(indent + line for line in text.split('\n'))

def repr_module(module) -> str:
    """Generate string representation of a module (no folding)"""
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = fansi(module.extra_repr(), 'blue')
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in module._modules.items():
        mod_str = repr_module(module)
        mod_str = indentify(mod_str, '  ')
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = module._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str

class TreeNode:
    """Node in the module tree structure"""
    def __init__(self, name: str, module_type: str, extra_info: str = "", parent=None):
        self.name = name
        self.module_type = module_type
        self.extra_info = extra_info
        self.children: List['TreeNode'] = []
        self.parent = parent
        self.expanded = True
        self.selected = False
        self.highlight_color = None

    def toggle_expand(self):
        """Toggle expand/collapse state"""
        self.expanded = not self.expanded

    def add_child(self, child: 'TreeNode'):
        """Add a child node"""
        self.children.append(child)
        child.parent = self

    def is_leaf(self) -> bool:
        """Check if node is a leaf (has no children)"""
        return len(self.children) == 0

def build_tree_from_module(module, parent: Optional[TreeNode] = None, name: str = "") -> TreeNode:
    """Convert PyTorch module to tree structure"""
    module_name = module._get_name()
    extra_info = module.extra_repr()
    
    if parent is None:
        # Root node
        node = TreeNode(name, module_name, extra_info)
    else:
        node = TreeNode(name, module_name, extra_info, parent)
        parent.add_child(node)
    
    # Add children
    for key, child_module in module._modules.items():
        build_tree_from_module(child_module, node, key)
    
    return node

class ModuleTreeExplorer:
    def __init__(self, module):
        self.module = module
        self.root = build_tree_from_module(module)
        self.cursor_y = 0
        self.scroll_offset = 0
        self.max_lines = 0
        self.visible_nodes: List[TreeNode] = []
        self.search_string = ""
        self.search_mode = False
        self.search_results: List[TreeNode] = []
        self.current_search_idx = -1
        self.status_message = ""
        self.screen = None
        self.max_y = 0
        self.max_x = 0
        self.color_pairs = {}
        
    def setup_colors(self):
        """Set up curses color pairs"""
        curses.start_color()
        curses.use_default_colors()
        
        self.color_pairs = {
            "normal": 0,
            "blue": 1,
            "green": 2,
            "yellow": 3,
            "red": 4,
            "cyan": 5,
            "magenta": 6,
            "selected": 7,
            "highlight": 8,
        }
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_BLUE, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        
    def flatten_visible_tree(self, node: TreeNode = None, depth: int = 0) -> List[Tuple[TreeNode, int]]:
        """Create flattened list of visible nodes with their depths"""
        if node is None:
            node = self.root
            
        result = [(node, depth)]
        
        if node.expanded:
            for child in node.children:
                result.extend(self.flatten_visible_tree(child, depth + 1))
                
        return result
    
    def update_visible_nodes(self):
        """Update the list of visible nodes"""
        flat_tree = self.flatten_visible_tree()
        self.visible_nodes = [node for node, _ in flat_tree]
        self.max_lines = len(self.visible_nodes)

    def render_tree(self):
        """Render the tree in the terminal"""
        self.screen.clear()
        self.update_visible_nodes()
        
        flat_tree = self.flatten_visible_tree()
        height, width = self.max_y, self.max_x
        
        # Adjust scroll if needed
        if self.cursor_y >= self.scroll_offset + height - 2:
            self.scroll_offset = self.cursor_y - height + 3
        elif self.cursor_y < self.scroll_offset:
            self.scroll_offset = self.cursor_y
            
        # Keep scroll offset in bounds
        self.scroll_offset = max(0, min(self.scroll_offset, max(0, len(flat_tree) - height + 2)))
        
        # Render visible portion of the tree
        visible_range = range(self.scroll_offset, min(self.scroll_offset + height - 2, len(flat_tree)))
        
        for i, idx in enumerate(visible_range):
            node, depth = flat_tree[idx]
            
            # Determine the prefix (tree branches)
            prefix = "  " * depth
            if node.children:
                prefix += "▼ " if node.expanded else "▶ "
            else:
                prefix += "  "
                
            # Determine node text
            if node.name:
                node_text = f"({node.name}): {node.module_type}"
            else:
                node_text = node.module_type
                
            if node.extra_info:
                if len(node.extra_info) > 30:  # Truncate long extra info
                    node_text += f"({node.extra_info[:30]}...)"
                else:
                    node_text += f"({node.extra_info})"
            
            # Determine attributes for rendering
            attrs = curses.A_NORMAL
            if idx == self.cursor_y:
                color = self.color_pairs["selected"]
                attrs |= curses.A_BOLD
            elif node.highlight_color:
                color = self.color_pairs[node.highlight_color]
            else:
                color = self.color_pairs["normal"]
                
            # Render the line
            line = f"{prefix}{node_text}"
            try:
                self.screen.addstr(i, 0, line[:width-1], curses.color_pair(color) | attrs)
            except curses.error:
                pass  # Ignore errors when writing to the last cell
        
        # Render status bar
        status_line = height - 1
        try:
            if self.search_mode:
                search_text = f"Search: {self.search_string}"
                self.screen.addstr(status_line, 0, search_text, curses.A_REVERSE)
                self.screen.clrtoeol()
            else:
                if self.status_message:
                    self.screen.addstr(status_line, 0, self.status_message, curses.A_REVERSE)
                    self.screen.clrtoeol()
                else:
                    help_text = "↑/↓:Navigate  Space:Toggle  f:Find  c:Color  q:Quit"
                    self.screen.addstr(status_line, 0, help_text, curses.A_REVERSE)
                    self.screen.clrtoeol()
        except curses.error:
            pass  # Ignore errors when writing to the last cell
            
        self.screen.refresh()
    
    def handle_key(self, key):
        """Handle keypress events"""
        if self.search_mode:
            return self.handle_search_key(key)
            
        if key == curses.KEY_UP:
            # Move cursor up
            self.cursor_y = max(0, self.cursor_y - 1)
        elif key == curses.KEY_DOWN:
            # Move cursor down
            self.cursor_y = min(self.max_lines - 1, self.cursor_y + 1)
        elif key == ord(' '):
            # Toggle expand/collapse
            if self.cursor_y < len(self.visible_nodes):
                node = self.visible_nodes[self.cursor_y]
                if not node.is_leaf():
                    node.toggle_expand()
        elif key == ord('f') or key == ord('/'):
            # Enter search mode
            self.search_mode = True
            self.search_string = ""
        elif key == ord('n'):
            # Jump to next search result
            self.goto_next_search_result()
        elif key == ord('c'):
            # Cycle highlight color of current node
            colors = ["blue", "green", "yellow", "red", "cyan", "magenta", None]
            if self.cursor_y < len(self.visible_nodes):
                node = self.visible_nodes[self.cursor_y]
                current_idx = colors.index(node.highlight_color) if node.highlight_color in colors else -1
                node.highlight_color = colors[(current_idx + 1) % len(colors)]
        elif key == ord('q'):
            # Quit
            return False
            
        return True
        
    def handle_search_key(self, key):
        """Handle keys while in search mode"""
        if key == 27:  # ESC
            # Exit search mode
            self.search_mode = False
            self.status_message = ""
        elif key == 10 or key == 13:  # Enter
            # Execute search
            self.search_mode = False
            self.execute_search()
        elif key == curses.KEY_BACKSPACE or key == 127:
            # Backspace
            self.search_string = self.search_string[:-1]
        elif 32 <= key <= 126:  # Printable ASCII
            # Add to search string
            self.search_string += chr(key)
            
        return True
        
    def execute_search(self):
        """Execute search with current search string"""
        if not self.search_string:
            self.status_message = ""
            return
            
        # Find all nodes matching the search string
        self.search_results = []
        pattern = re.compile(self.search_string, re.IGNORECASE)
        
        def search_node(node):
            text = f"{node.name} {node.module_type} {node.extra_info}"
            if pattern.search(text):
                self.search_results.append(node)
                
            for child in node.children:
                search_node(child)
                
        search_node(self.root)
        
        # Jump to first result
        if self.search_results:
            self.current_search_idx = 0
            self.jump_to_node(self.search_results[0])
            self.status_message = f"Found {len(self.search_results)} matches"
        else:
            self.status_message = f"No matches for '{self.search_string}'"
            
    def goto_next_search_result(self):
        """Go to the next search result"""
        if not self.search_results:
            self.status_message = "No search results"
            return
            
        self.current_search_idx = (self.current_search_idx + 1) % len(self.search_results)
        self.jump_to_node(self.search_results[self.current_search_idx])
        self.status_message = f"Match {self.current_search_idx + 1}/{len(self.search_results)}"
        
    def jump_to_node(self, target_node):
        """Jump to a specific node in the tree"""
        # Ensure path to node is expanded
        node = target_node
        while node.parent:
            node.parent.expanded = True
            node = node.parent
            
        # Update visible nodes
        self.update_visible_nodes()
        
        # Find node index
        try:
            idx = self.visible_nodes.index(target_node)
            self.cursor_y = idx
        except ValueError:
            pass  # Node not found (shouldn't happen)
        
    def run(self, stdscr):
        """Main application loop"""
        self.screen = stdscr
        curses.curs_set(0)  # Hide cursor
        self.setup_colors()
        
        # Get terminal dimensions
        self.max_y, self.max_x = self.screen.getmaxyx()
        
        running = True
        while running:
            self.render_tree()
            key = self.screen.getch()
            running = self.handle_key(key)
            
            # Update dimensions if terminal resized
            new_y, new_x = self.screen.getmaxyx()
            if new_y != self.max_y or new_x != self.max_x:
                self.max_y, self.max_x = new_y, new_x
                self.screen.clear()

def explore_module(module):
    """Start the interactive module explorer"""
    explorer = ModuleTreeExplorer(module)
    curses.wrapper(explorer.run)

if __name__ == "__main__":
    # Load the model
    model = diffusers.CogVideoXTransformer3DModel()
    
    # Run the interactive explorer
    explore_module(model)