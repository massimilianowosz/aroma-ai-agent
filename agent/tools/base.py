from typing import List, Any, Callable
import inspect

class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool
        description (str): A textual description of what the tool does
        func (callable): The function this tool wraps
        arguments (list): A list of (argument_name, argument_type) tuples
        outputs (str): The return type of the wrapped function
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: Callable, 
                 arguments: List[tuple],
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """Return a string representation of the tool."""
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Arguments: {args_str}\n"
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs) -> Any:
        """Invoke the underlying function with provided arguments."""
        return self.func(*args, **kwargs)

def tool(func: Callable) -> Tool:
    """
    A decorator that creates a Tool instance from the given function.
    
    Args:
        func (callable): The function to wrap as a tool
        
    Returns:
        Tool: A Tool instance wrapping the provided function
    """
    signature = inspect.signature(func)
    
    arguments = []
    for param in signature.parameters.values():
        annotation = param.annotation
        annotation_name = (
            annotation.__name__ 
            if hasattr(annotation, '__name__') 
            else str(annotation).replace('typing.', '')
        )
        arguments.append((param.name, annotation_name))
    
    return_annotation = signature.return_annotation
    outputs = (
        "Any" if return_annotation is inspect._empty
        else return_annotation.__name__ if hasattr(return_annotation, '__name__')
        else str(return_annotation).replace('typing.', '')
    )
    
    return Tool(
        name=func.__name__,
        description=func.__doc__ or "No description provided.",
        func=func,
        arguments=arguments,
        outputs=outputs
    )