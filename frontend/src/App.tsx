import React, { useState, useRef, useEffect } from 'react';
import {
    AppBar, Toolbar, Typography, Paper, TextField, Button,
    Box, Container, IconButton, Divider, Card, CardContent,
    List, ListItem, ListItemText, CircularProgress, Link,
    Tooltip
} from '@mui/material';
import {
    ChatBubble as ChatIcon,
    Refresh as RefreshIcon,
    Send as SendIcon,
    Link as LinkIcon,
    FormatQuote as QuoteIcon,
    RestartAlt as RestartIcon
} from '@mui/icons-material';

interface Message {
    type: 'user' | 'assistant';
    content: string;
    tools_used?: any[];
    tool_results?: {
        useful_links?: {
            title: string;
            link: string;
        }[];
        quotes?: string[];
    };
}

const App: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom when messages change
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        // Add user message to chat
        const userMessage: Message = {
            type: 'user',
            content: query
        };
        setMessages([...messages, userMessage]);

        // Clear input and set loading
        setQuery('');
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                // Include the conversationId in the request if we have one
                body: JSON.stringify({
                    query: query.trim(),
                    conversation_id: conversationId
                }),
                // Important for cookies to be sent and received
                credentials: 'include'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get response');
            }

            const data = await response.json();

            // Save the conversation ID if it's in the response
            if (data.conversation_id) {
                setConversationId(data.conversation_id);
            }

            // Extract the actual content - handle both nested and flat structures
            const content = typeof data.content === 'object' && data.content.content
                ? data.content.content
                : data.content || 'No response received';

            // Get tool results from the right place
            const toolResults = data.content && data.content.tool_results
                ? data.content.tool_results
                : data.tool_results || {};

            // Get tools used from the right place
            const toolsUsed = data.content && data.content.tools_used
                ? data.content.tools_used
                : data.tools_used || [];

            // Add assistant message with all the data
            const assistantMessage: Message = {
                type: 'assistant',
                content: content,
                tools_used: toolsUsed,
                tool_results: toolResults
            };

            setMessages(prevMessages => [...prevMessages, assistantMessage]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const clearChat = () => {
        setMessages([]);
        setError(null);
    };

    const startNewConversation = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/new-conversation', {
                method: 'POST',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to create new conversation');
            }

            const data = await response.json();
            setConversationId(data.conversation_id);
            setMessages([]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to start new conversation');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#f5f5f5' }}>
            {/* Header */}
            <AppBar position="static" color="default" elevation={1}>
                <Toolbar>
                    <ChatIcon sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Financial News Chat
                    </Typography>
                    <Tooltip title="Reset conversation (maintains message history)">
                        <IconButton onClick={clearChat} size="small" color="inherit" sx={{ mr: 1 }}>
                            <RefreshIcon />
                            <Typography variant="caption" sx={{ ml: 1 }}>Clear History</Typography>
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Start a completely new conversation">
                        <IconButton onClick={startNewConversation} size="small" color="primary">
                            <RestartIcon />
                            <Typography variant="caption" sx={{ ml: 1 }}>New Conversation</Typography>
                        </IconButton>
                    </Tooltip>
                </Toolbar>
            </AppBar>

            {/* Chat messages container */}
            <Container maxWidth="lg" sx={{ flex: 1, overflow: 'auto', py: 2 }}>
                {messages.length === 0 ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '70vh', color: 'text.secondary' }}>
                        <ChatIcon sx={{ fontSize: 60, mb: 2, color: 'text.disabled' }} />
                        <Typography variant="h6">Welcome to Financial News Chat</Typography>
                        <Typography variant="body2">Ask questions about financial news, market trends, or company information</Typography>
                        {conversationId && (
                            <Typography variant="caption" sx={{ mt: 2, color: 'text.disabled' }}>
                                Conversation ID: {conversationId}
                            </Typography>
                        )}
                    </Box>
                ) : (
                    <List>
                        {messages.map((message, index) => (
                            <ListItem
                                key={index}
                                sx={{
                                    display: 'flex',
                                    justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
                                    mb: 2,
                                    px: 0
                                }}
                            >
                                <Card
                                    elevation={1}
                                    sx={{
                                        maxWidth: '75%',
                                        bgcolor: message.type === 'user' ? 'primary.main' : 'background.paper',
                                        color: message.type === 'user' ? 'white' : 'text.primary'
                                    }}
                                >
                                    <CardContent>
                                        {message.type === 'user' ? (
                                            <Typography>{message.content}</Typography>
                                        ) : (
                                            <Box>
                                                <Box>
                                                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>
                                                        {message.content}
                                                    </Typography>
                                                </Box>

                                                {/* This is the key change: using Boolean() to prevent "0" from rendering */}
                                                {Boolean(message.tool_results?.useful_links?.length || message.tool_results?.quotes?.length) && (
                                                    <Box sx={{ mt: 2 }}>
                                                        <Divider sx={{ my: 2 }} />

                                                        {/* Links */}
                                                        {message.tool_results?.useful_links && message.tool_results.useful_links.length > 0 && (
                                                            <Box sx={{ mb: 2 }}>
                                                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, color: 'primary.main' }}>
                                                                    <LinkIcon fontSize="small" sx={{ mr: 0.5 }} />
                                                                    <Typography variant="subtitle2">Useful Links</Typography>
                                                                </Box>
                                                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                                    {message.tool_results.useful_links.map((link, i) => (
                                                                        <Link
                                                                            key={i}
                                                                            href={link.link}
                                                                            target="_blank"
                                                                            rel="noreferrer"
                                                                            underline="hover"
                                                                            sx={{
                                                                                display: 'block',
                                                                                p: 1,
                                                                                bgcolor: 'action.hover',
                                                                                borderRadius: 1,
                                                                                fontSize: '0.875rem'
                                                                            }}
                                                                        >
                                                                            {link.title}
                                                                        </Link>
                                                                    ))}
                                                                </Box>
                                                            </Box>
                                                        )}

                                                        {/* Quotes */}
                                                        {message.tool_results?.quotes && message.tool_results.quotes.length > 0 && (
                                                            <Box>
                                                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, color: 'success.main' }}>
                                                                    <QuoteIcon fontSize="small" sx={{ mr: 0.5 }} />
                                                                    <Typography variant="subtitle2">Supporting Quotes</Typography>
                                                                </Box>
                                                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                                    {message.tool_results.quotes.map((quote, i) => (
                                                                        <Box
                                                                            key={i}
                                                                            sx={{
                                                                                p: 1.5,
                                                                                bgcolor: 'background.default',
                                                                                borderLeft: '4px solid',
                                                                                borderColor: 'success.light',
                                                                                borderRadius: 1,
                                                                                fontSize: '0.875rem'
                                                                            }}
                                                                        >
                                                                            "{quote}"
                                                                        </Box>
                                                                    ))}
                                                                </Box>
                                                            </Box>
                                                        )}
                                                    </Box>
                                                )}
                                            </Box>
                                        )}
                                    </CardContent>
                                </Card>
                            </ListItem>
                        ))}

                        {/* Loading indicator */}
                        {isLoading && (
                            <ListItem sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2, px: 0 }}>
                                <Card elevation={1} sx={{ p: 2 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                        <CircularProgress size={20} />
                                        <Typography variant="body2" sx={{ ml: 2 }}>Thinking...</Typography>
                                    </Box>
                                </Card>
                            </ListItem>
                        )}

                        {/* Error message */}
                        {error && (
                            <ListItem sx={{ display: 'flex', justifyContent: 'center', mb: 2, px: 0 }}>
                                <Paper elevation={0} sx={{ p: 2, bgcolor: '#FFEBEE', borderLeft: '4px solid', borderColor: 'error.main' }}>
                                    <Typography color="error">{error}</Typography>
                                </Paper>
                            </ListItem>
                        )}

                        <div ref={messagesEndRef} />
                    </List>
                )}
            </Container>

            {/* Input form */}
            <Paper
                component="form"
                onSubmit={handleSubmit}
                elevation={3}
                sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}
            >
                <Box sx={{ display: 'flex', maxWidth: 'lg', mx: 'auto' }}>
                    <TextField
                        fullWidth
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask about financial news, market trends, or stocks..."
                        disabled={isLoading}
                        variant="outlined"
                        size="small"
                        sx={{ mr: 2 }}
                    />
                    <Button
                        type="submit"
                        variant="contained"
                        disabled={isLoading || !query.trim()}
                        endIcon={<SendIcon />}
                    >
                        Ask
                    </Button>
                </Box>
            </Paper>
        </Box>
    );
};

export default App;
