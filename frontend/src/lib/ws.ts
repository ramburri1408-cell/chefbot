// frontend/src/lib/ws.ts
//
// WebSocket client with automatic reconnection and exponential backoff.
//
// Why not just `new WebSocket()`?
// - Native WebSocket doesn't reconnect on drop — mobile networks drop
//   constantly. Without reconnect, the user's chat session just silently dies.
// - Session continuity: we store the session_id received from the server and
//   send it on reconnect so the API pod can restore state from Redis.

import { WSEvent } from '@/types'

type MessageHandler = (event: WSEvent) => void
type StatusHandler  = (status: 'connected' | 'disconnected' | 'reconnecting') => void

const BASE_DELAY  = 500
const MAX_DELAY   = 30_000
const MAX_RETRIES = 10

export class ChatSocket {
  private ws:         WebSocket | null = null
  private url:        string
  private sessionId:  string | null    = null
  private retries:    number           = 0
  private destroyed:  boolean          = false
  private onMessage:  MessageHandler
  private onStatus:   StatusHandler

  constructor(url: string, onMessage: MessageHandler, onStatus: StatusHandler) {
    this.url       = url
    this.onMessage = onMessage
    this.onStatus  = onStatus
    this.connect()
  }

  private connect() {
    if (this.destroyed) return

    this.ws = new WebSocket(
      this.sessionId ? `${this.url}?session_id=${this.sessionId}` : this.url
    )

    this.ws.onopen = () => {
      this.retries = 0
      this.onStatus('connected')
    }

    this.ws.onmessage = (e) => {
      try {
        const event: WSEvent = JSON.parse(e.data)
        if (event.type === 'session' && event.session_id) {
          this.sessionId = event.session_id
        }
        this.onMessage(event)
      } catch {
        console.error('WS parse error', e.data)
      }
    }

    this.ws.onclose = () => {
      if (!this.destroyed) {
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  private scheduleReconnect() {
    if (this.retries >= MAX_RETRIES) {
      this.onStatus('disconnected')
      return
    }
    this.onStatus('reconnecting')
    const delay = Math.min(BASE_DELAY * 2 ** this.retries, MAX_DELAY)
    this.retries++
    setTimeout(() => this.connect(), delay)
  }

  send(message: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ message }))
    }
  }

  destroy() {
    this.destroyed = true
    this.ws?.close()
  }
}
