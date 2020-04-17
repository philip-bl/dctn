from typing import Tuple, Union, Hashable, Dict

import opt_einsum as oe
from opt_einsum.contract import ContractExpression

from torch import Tensor

from .singleton import Singleton


ContractArgs = Tuple[Union[Hashable, Tensor], ...]  # args you call oe.contract on
ContractExpressionArgs = Tuple[Hashable, ...]  # args you call oe.contract_expression on


def tensors_to_shapes(*args: ContractArgs) -> ContractExpressionArgs:
    return tuple(x.shape if isinstance(x, Tensor) else x for x in args)


class ContractionPathCache(metaclass=Singleton):
    def __init__(self):
        self.paths: Dict[ContractExpressionArgs, ContractExpression] = {}

    def contract(self, *args: ContractArgs) -> Tensor:
        contract_expression_args = tensors_to_shapes(*args)
        if contract_expression_args not in self.paths:
            self.paths[contract_expression_args] = oe.contract_expression(
                *contract_expression_args, optimize="auto-hq"
            )
        return self.paths[contract_expression_args](
            *(x for x in args if isinstance(x, Tensor))
        )


def contract(*args: ContractArgs) -> Tensor:
    return ContractionPathCache().contract(*args)
