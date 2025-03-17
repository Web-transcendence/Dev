import {z} from "zod";

export const profileSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres")
});


export const signUpSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});

export const signInSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});

export * from "./schema.js";