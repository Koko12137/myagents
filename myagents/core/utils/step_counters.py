from uuid import uuid4
from typing import Union
from asyncio import Lock

from loguru import logger
from myagents.core.interface import StepCounter, Logger
from myagents.core.messages import CompletionUsage


class MaxStepsError(Exception):
    """MaxStepsError is the error raised when the max steps is reached.
    
    Attributes:
        current (int):
            The current step of the step counter.
        limit (int):
            The limit of the step counter.
    """
    current: int
    limit: int
    
    def __init__(self, current: int, limit: int) -> None:
        """Initialize the MaxStepsError.
        
        Args:
            current (int):
                The current step of the step counter.
            limit (int):
                The limit of the step counter.
        """
        self.current = current
        self.limit = limit
        
    def __str__(self) -> str:
        """Return the string representation of the MaxStepsError.
        
        Returns:
            str:
                The string representation of the MaxStepsError.
        """
        return f"Max auto steps reached. Current: {self.current}, Limit: {self.limit}"


class MaxStepCounter(StepCounter):
    """MaxStepCounter allows the user to set the limit of the step counter, and the limit will **never** be reset. 
    
    Attributes:
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger, defaults to logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    lock: Lock
    custom_logger: Logger
    
    def __init__(self, limit: int = 10, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10):
                The limit of the step counter. 
        """
        self.uid = uuid4().hex
        self.limit = limit
        self.current = 0
        self.custom_logger = custom_logger
        self.lock = Lock()
        
    async def check_limit(self) -> bool:
        """Check if the limit of the step counter is reached.
        
        Returns:
            bool:
                Whether the limit of the step counter is reached.
                
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        if self.current >= self.limit:
            e = MaxStepsError(self.current, self.limit)
            self.custom_logger.error(e)
            raise e
        return False
        
    async def step(self, step: Union[int, float] = 1) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (Union[int, float], optional, defaults to 1):
                The step to increment. 
                
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        async with self.lock:
            self.current += 1
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            e = MaxStepsError(self.current, self.limit)
            # The max steps error is raised, then update the task status to cancelled
            self.custom_logger.error(e)
            raise e
        
    async def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        raise NotImplementedError("Reset is not supported for the max step counter.")
    
    async def update_limit(self, limit: Union[int, float]) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        raise NotImplementedError("Update limit is not supported for the max step counter.")
    
    async def recharge(self, limit: Union[int, float]) -> None:
        """Recharge the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        raise NotImplementedError("Recharge is not supported for the max step counter.")


class BaseStepCounter(MaxStepCounter):
    """BaseStepCounter count only the action steps, without any concern of the token usage. The step counter could be 
    reset by the user. 
    
    Attributes:
        uid (str):
            The unique identifier of the step counter. 
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    custom_logger: Logger
    
    def __init__(self, limit: int = 10, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10):
                The limit of the step counter. 
        """
        super().__init__(limit, custom_logger)
        
    async def step(self, step: CompletionUsage) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (CompletionUsage):
                The step to increment. 
                
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        async with self.lock:
            self.current += 1
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            # Request the user to reset the step counter
            reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {e.limit} steps? (y/n)")
            
            if reset == "y":
                # Reset the step counter and continue the loop
                await self.reset()
            else:
                e = MaxStepsError(self.current, self.limit)
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(e)
                raise e
        
    async def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        async with self.lock:
            self.current = 0
        
    async def update_limit(self, limit: int) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (int):
                The limit of the step counter. 
        """
        async with self.lock:
            self.limit = limit
        
    async def recharge(self, limit: Union[int, float]) -> None:
        """Recharge the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        async with self.lock:
            self.limit += limit


class TokenStepCounter(BaseStepCounter):
    """TokenStepCounter counts the token usage of the LLM. The step counter will ask for reset when the token usage is greater than the limit. 
    
    Attributes:
        uid (str):
            The unique identifier of the step counter. 
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    custom_logger: Logger
    
    def __init__(self, limit: int = 10000, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10000):
                The limit of the step counter. Default to 10 thousand. 
            custom_logger (Logger, optional, defaults to logger):
                The custom logger to use for the step counter. 
        """
        super().__init__(limit, custom_logger)
    
    async def step(self, step: CompletionUsage) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (CompletionUsage):
                The step to increment. 
        """
        async with self.lock:
            self.current += step.total_tokens
            self.custom_logger.warning(f"The current Token Usage is {self.current}, the Limit is {self.limit}.")
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            # Request the user to reset the step counter
            reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {self.limit} steps? (y/n)")
            
            if reset == "y":
                # Reset the step counter and continue the loop  
                await self.reset()
            else:
                e = MaxStepsError(self.current, self.limit)
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(e)
                raise e
            
    async def reset(self) -> None:
        """Reset is not supported for the token step counter.
        """
        raise NotImplementedError("Reset is not supported for the token step counter. Please use `recharge` to update the limit.")
