import {z} from "zod";

export const profileSchema = z.object({
    id: z.string().regex(/^\d+$/, "Only numeric characters are allowed")
});

export const manageFriendSchema = z.object({
    friendNickName: z.string().min(3, "Minimum 3 caracteres")
})

export const tournamentIdSchema = z.object({
    tournamentId: z.number()
})

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

export const idListSchema = z.object({
    ids: z.array(z.string()).transform((arr: string[]) =>
        arr.map((idStr: string) => {
            const num = Number(idStr);
            if (isNaN(num)) throw new Error(`Invalid number: ${idStr}`);
            return num;
        })
    )
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

export const notifySchema = z.object({
    ids: z.array(z.string()).transform((arr: string[]) =>
        arr.map((idStr: string) => {
            const num = Number(idStr);
            if (isNaN(num)) throw new Error(`Invalid number: ${idStr}`);
            return num;
        })
    ),
    event: z.string(),
    body: z.string().transform((val) => {
        return JSON.parse(val);
    })
})

export * from "./schema.js";