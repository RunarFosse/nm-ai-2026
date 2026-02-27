from typing import TypedDict, NotRequired, List

class Grid(TypedDict):
   """ Grid Schema Definition """
   width: int
   height: int
   walls: List[List[int]]

class Bot(TypedDict):
   """ Bot Schema Definition """
   id: int
   position: List[int]
   inventory: List[str]

class Item(TypedDict):
   """ Item Schema Definition """
   id: str
   type: str
   position: List[int]

class Order(TypedDict):
   """ Order Schema Definition """
   id: str
   items_required: List[str]
   items_delivered: List[str]
   complete: bool
   status: str

class State(TypedDict):
    """ State Schema Definition """ 
    type: str
    round: int
    max_rounds: int
    grid: Grid
    bots: List[Bot]
    items: List[Item]
    orders: List[Order]
    drop_off: List[int]
    score: int

class Action(TypedDict):
   """ Action Schema Definition """
   bot: int
   action: str
   item_id: NotRequired[str]

class Response(TypedDict):
   """ Response Schema Definition """
   actions: List[Action]
