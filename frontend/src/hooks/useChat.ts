// frontend/src/hooks/useChat.ts
//
// Central hook for the entire chat experience.
// Manages WebSocket lifecycle, message accumulation during streaming,
// and dish card injection.
//
// Design: streaming tokens are accumulated into the current "streaming"
// message rather than creating a new message per token — the UI re-renders
// with each token but the message count stays stable.

import { useEffect, useRef, useState, useCallback } from 'react'
import { v4 as uuid } from 'uuid'
import { ChatSocket } from '@/lib/ws'
import { Dish, Message, WSEvent } from '@/types'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws/chat'

export function useChat() {
  const [messages,  setMessages]  = useState<Message[]>([])
  const [status,    setStatus]    = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected')
  const [isThinking, setIsThinking] = useState(false)

  const socketRef       = useRef<ChatSocket | null>(null)
  const streamingIdRef  = useRef<string | null>(null)
  const pendingDishesRef = useRef<Dish[] | null>(null)

  useEffect(() => {
    const socket = new ChatSocket(
      WS_URL,
      handleEvent,
      setStatus,
    )
    socketRef.current = socket
    return () => socket.destroy()
  }, [])

  const handleEvent = useCallback((event: WSEvent) => {
    switch (event.type) {
      case 'session':
        // session_id stored in ChatSocket — nothing to do in UI
        break

      case 'token': {
        const token = event.content as string
        if (!streamingIdRef.current) {
          // First token — create a new streaming message
          const id = uuid()
          streamingIdRef.current = id
          setIsThinking(false)
          setMessages(prev => [
            ...prev,
            {
              id,
              role:      'assistant',
              content:   token,
              dishes:    pendingDishesRef.current ?? undefined,
              timestamp: Date.now(),
              streaming: true,
            },
          ])
          pendingDishesRef.current = null
        } else {
          // Append to existing streaming message
          const id = streamingIdRef.current
          setMessages(prev =>
            prev.map(m =>
              m.id === id ? { ...m, content: m.content + token } : m
            )
          )
        }
        break
      }

      case 'dishes': {
        // Dishes arrive before tokens — hold until first token
        pendingDishesRef.current = event.content as Dish[]
        break
      }

      case 'done': {
        // Mark streaming complete
        const id = streamingIdRef.current
        if (id) {
          setMessages(prev =>
            prev.map(m =>
              m.id === id ? { ...m, streaming: false } : m
            )
          )
        }
        streamingIdRef.current = null
        setIsThinking(false)
        break
      }

      case 'error': {
        setIsThinking(false)
        streamingIdRef.current = null
        setMessages(prev => [
          ...prev,
          {
            id:        uuid(),
            role:      'assistant',
            content:   event.content as string || 'Something went wrong — please try again.',
            timestamp: Date.now(),
          },
        ])
        break
      }
    }
  }, [])

  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || status !== 'connected') return

    setMessages(prev => [
      ...prev,
      {
        id:        uuid(),
        role:      'user',
        content:   text.trim(),
        timestamp: Date.now(),
      },
    ])
    setIsThinking(true)
    socketRef.current?.send(text)
  }, [status])

  return { messages, status, isThinking, sendMessage }
}
