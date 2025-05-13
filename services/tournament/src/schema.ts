import {z} from "zod";


export const tournamentIdSchema = z.object({
    tournamentId: z.number()
})

export * from "./schema.js";