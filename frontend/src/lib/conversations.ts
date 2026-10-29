import { useEffect, useState } from "react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: boolean;
  createdAt: number;
  durationMs?: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

const STORAGE_KEY = "lab-agent:conversations";

export const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

export function newConversation(): Conversation {
  return { id: uid(), title: "New conversation", messages: [], updatedAt: Date.now() };
}

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* storage unavailable or corrupt */
  }
  return [];
}

/** Conversations live only in the browser: the /chat endpoint is stateless. */
export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const stored = load();
    return stored.length ? stored : [newConversation()];
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.filter((c) => c.messages.length)));
    } catch {
      /* storage full or unavailable */
    }
  }, [conversations]);

  return [conversations, setConversations] as const;
}
