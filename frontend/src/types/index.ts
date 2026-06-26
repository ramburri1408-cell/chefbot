// frontend/src/types/index.ts

export interface Dish {
  name: string
  category: string
  calories: number
  protein_g: number
  carbs_g?: number
  fat_g?: number
  price: number
  dietary: string[]
  description: string
  similarity_score?: number
}

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  dishes?: Dish[]
  timestamp: number
  streaming?: boolean
}

export type WSEventType = 'session' | 'token' | 'dishes' | 'done' | 'error'

export interface WSEvent {
  type: WSEventType
  content?: string | Dish[]
  session_id?: string
}
