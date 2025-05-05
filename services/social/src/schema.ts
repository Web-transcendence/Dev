import {z} from "zod";

export const manageFriendSchema = z.object({
    friendNickName: z.string().min(3, "Minimum 3 caracteres")
})

export * from "./schema.js";