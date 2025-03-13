import Database from "better-sqlite3";
import {FastifyReply, FastifyRequest} from "fastify";
import {z} from "zod";
import sanitizeHtml from "sanitize-html";
import bcrypt from "bcrypt";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

// --     DELETE FROM Client; -- remove olds clients and reset database, remove for production
Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);

export function emailExist(email: string) {
    const existingClient = Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email);
    return !!existingClient;
}

const clientSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});


export async function createClient(req: FastifyRequest, res: FastifyReply) {
    const mystuff = req.body;
    const zod_result = clientSchema.safeParse(mystuff);
    if (!zod_result.success)
        return res.status(400).send({json: zod_result.error.format()});
    const { data } = zod_result;
    let {name, email, password} = data;
    name = sanitizeHtml(name); // Protection from XSS attacks
    email = sanitizeHtml(email);
    password = sanitizeHtml(password);
    console.log(name, email, password);
    if (!name || !email || !password)
        return res.status(454).send({error: "All information are required !"});
    const hashedPassword = await bcrypt.hash(password, 10);

    const insert = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)");
    const result = insert.run(name, email, hashedPassword);
    // const rows = Client_db.prepare(`SELECT * FROM Client`).all(); // Print clients info
    // console.table(rows);
    if (result.changes === 1) {
        Client_db.prepare("SELECT * FROM Client");
        const newClientId = result.lastInsertRowid;
        const newClient = Client_db.prepare("SELECT name FROM Client WHERE id = ?").get(newClientId) as { name: string } | undefined;
        if (newClient) {
            console.log("New client registred:", newClient.name);
        }
        return res.status(201).send({ id: result.lastInsertRowid });
    } else {
        return res.status(500).send({ error: "Error from database " });
    }
}