import {z} from "zod";

export const verifySchema = z.object({
    secret: z.string().regex(/^\d{6}$/, {
        message: "only 6 digits are allowed",
    }),
    nickName: z.string().min(3, "Minimum 3 caracteres")
})

export const pictureSchema = z.object({
    pictureURL: z.string().regex(
        /^data:image\/(jpeg|png);base64/,
        'String must start with "data:image/jpeg;base64"'
    )
})

export const idArraySchema = z.object({
        ids: z.array(z.number())
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

export const passwordSchema = z.object({
    password: z.string().min(6, "Minimum 6 caracteres")
})

export const nickNameSchema = z.object({
    nickName: z.string().min(3, "Minimum 3 caracteres")
})

export const notifySchema = z.object({
    ids: z.array(z.number()),
    event: z.string(),
    data: z.any()
})

export * from "./schema.js";