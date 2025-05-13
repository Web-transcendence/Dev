import {z} from "zod";

export const manageFriendSchema = z.object({
    friendNickName: z.string().min(3, "3 character or more for the nickname")
        .regex(/^[a-zA-Z0-9]+$/, "only alphanumeric character accepted for the nickname"),
})

export const checkFriendSchema = z.object({
    id1: z.number(),
    id2: z.number()
})

export * from "./schema.js";