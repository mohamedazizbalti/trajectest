from langchain_core.tools import tool
from langgraph.types import interrupt

REFUND_APPROVAL_LIMIT_EUROS = 100

ORDERS = {
    "1041": {"customer": "Lina", "item": "Wireless earbuds", "price_euros": 59, "status": "delivered"},
    "1042": {"customer": "Omar", "item": "Espresso machine", "price_euros": 249, "status": "delivered"},
    "1043": {"customer": "Sara", "item": "Running shoes", "price_euros": 120, "status": "shipped"},
}


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by its id. Returns the customer, item, price in euros and status."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {"order_id": order_id, **order}


@tool
def issue_refund(order_id: str, amount_euros: float) -> str:
    """Refund an order. Refunds above the approval limit pause for a human reviewer."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"No order found with id {order_id}"
    if amount_euros > order["price_euros"]:
        return f"Refund of €{amount_euros:g} is more than the order price of €{order['price_euros']}"

    if amount_euros > REFUND_APPROVAL_LIMIT_EUROS:
        decision = interrupt(
            {
                "type": "refund_approval",
                "message": f"Refund of €{amount_euros:g} for order {order_id} needs human approval",
                "order_id": order_id,
                "amount_euros": amount_euros,
            }
        )
        if not decision.get("approved"):
            return f"Refund of €{amount_euros:g} for order {order_id} was rejected by a human reviewer"

    return f"Refunded €{amount_euros:g} for order {order_id}"


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that has not been delivered yet."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"No order found with id {order_id}"
    if order["status"] == "delivered":
        return f"Order {order_id} is already delivered and cannot be cancelled; offer a refund instead"
    return f"Cancelled order {order_id}"


@tool
def escalate_to_human(reason: str) -> str:
    """Hand the conversation to a human support agent when the request is outside the tools' scope."""
    return f"Escalated to a human support agent: {reason}"


SUPPORT_TOOLS = [get_order, issue_refund, cancel_order, escalate_to_human]
