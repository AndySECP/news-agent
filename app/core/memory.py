# conversational memory manager


class ConversationManager:
    """Simple conversation history manager"""

    def __init__(self, max_turns=5, max_tokens=4000):
        """
        Initialize the conversation manager

        Args:
            max_turns: Maximum number of conversation turns to store
            max_tokens: Maximum total tokens in history before summarization
        """
        self.conversations = {}  # Dict of conversation_id -> history
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def add_exchange(self, conversation_id, query, response):
        """
        Add a query-response pair to the conversation history

        Args:
            conversation_id: Unique ID for the conversation
            query: The user's query
            response: The system's response
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # Add the new exchange
        self.conversations[conversation_id].append(
            {"query": query, "response": response}
        )

        # Trim to max_turns
        if len(self.conversations[conversation_id]) > self.max_turns:
            self.conversations[conversation_id] = self.conversations[conversation_id][
                -self.max_turns :
            ]

    def get_conversation_messages(self, conversation_id):
        """
        Get conversation history formatted as messages for the LLM
        
        Args:
            conversation_id: Unique ID for the conversation
            
        Returns:
            List of message objects for the LLM
        """
        if conversation_id not in self.conversations:
            return []
            
        messages = []
        for exchange in self.conversations[conversation_id]:
            messages.append({"role": "user", "content": exchange["query"]})
            
            # Extract the full response context, not just content
            response_content = exchange["response"].get("content", "")
            
            # Add structured data from tool results if available
            if "tool_results" in exchange["response"]:
                tool_results = exchange["response"]["tool_results"]
                if "useful_links" in tool_results:
                    response_content += "\n\nUseful Links:\n"
                    for link in tool_results["useful_links"]:
                        response_content += f"- {link['title']}\n"
                        
                if "quotes" in tool_results:
                    response_content += "\n\nQuotes:\n"
                    for quote in tool_results["quotes"]:
                        response_content += f"- {quote}\n"
                        
            messages.append({"role": "assistant", "content": response_content})
            
        return messages

    def get_last_response(self, conversation_id):
        """
        Get the most recent response for a given conversation ID
        
        Args:
            conversation_id: Unique ID for the conversation
            
        Returns:
            The most recent response, or None if no conversation history exists
        """
        if conversation_id not in self.conversations or not self.conversations[conversation_id]:
            return None
            
        # Return the response from the last exchange
        return self.conversations[conversation_id][-1].get("response")
