import {z} from "zod";

export const profileSchema = z.object({
    id: z.string().regex(/^\d+$/, "Only numeric characters are allowed")
});

export const manageFriendSchema = z.object({
    friendNickName: z.string().min(3, "Minimum 3 caracteres")
})

export const verifySchema = z.object({
    secret: z.number().int().gte(100000).lte(999999)
})

export const signUpSchema = z.object({
    nickName: z.string().min(3, "Minimum 3 caracteres"),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});

export const signInSchema = z.object({
    nickName: z.string().min(3, "Minimum 3 caracteres"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});

export * from "./schema.js";