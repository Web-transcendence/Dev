import { FastifyInstance } from 'fastify';
import Database from 'better-sqlite3';
import sanitizeHtml from "sanitize-html";
import bcrypt from "bcrypt";
import {z} from "zod";

export const Trans_Database = new Database('database.db')  // Importation de sqlite

Trans_Database.exec(`
    DELETE FROM clients; -- remove olds clients and reset database, remove for production
    DELETE FROM sqlite_sequence; -- same
    CREATE TABLE IF NOT EXISTS clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);

const clientSchema = z.object({
    name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
    email: z.string().email("Email invalide"),
    password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
});

export async function CreateClient(fastify: FastifyInstance) {
    fastify.post("/post/create", async (request, reply) => {
        try {
            const mystuff = request.body;
            const zod_result = clientSchema.safeParse(mystuff);
            if (!zod_result.success) {
                return zod_result.error.format();
            }
            const { data } = zod_result;
            let {name, email, password} = data;
            name = sanitizeHtml(name); // Protection from XSS attacks
            email = sanitizeHtml(email);
            password = sanitizeHtml(password);
            if (!name || !email || !password) {
                return reply.status(454).send({error: "All information are required !"});
            }
            const existingClient = Trans_Database.prepare("SELECT * FROM clients WHERE email = ?").get(email);
            if (existingClient) {
                return reply.status(431).send({error: "Email already in use"});
            }
            const hashedPassword = await bcrypt.hash(password, 10);
            const insert = Trans_Database.prepare("INSERT INTO clients (name, email, password) VALUES (?, ?, ?)");
            const result = insert.run(name, email, hashedPassword);
            const rows = Trans_Database.prepare(`SELECT * FROM clients`).all();
            console.table(rows);
            if (result.changes === 1) {
                // const table = Trans_Database.prepare("SELECT * FROM clients");
                // console.table(table.database);
                const newClientId = result.lastInsertRowid;
                const newClient = Trans_Database.prepare("SELECT name FROM clients WHERE id = ?").get(newClientId) as {
                    name: string
                } | undefined;
                if (newClient) {
                    console.log("New client registered:", newClient.name);
                    return reply.status(201).send({redirect: `/part/login?name=${encodeURIComponent(newClient?.name || "User")}`});
                }
            } else
                return reply.status(500).send({error: "Error when creating client acount"});
        } catch (error) {
            console.error("Error when register:", error);
            return reply.status(500).send({error: "Error from server"});
        }
    });
}