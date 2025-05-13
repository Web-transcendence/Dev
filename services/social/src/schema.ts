import {z} from "zod";

export const manageFriendSchema = z.object({
    friendNickName: z.string().min(3, "Minimum 3 caracteres")
})

export const checkFriendSchema = z.object({
    id1: z.number(),
    id2: z.number()
})

export * from "./schema.js";