import Database from "better-sqlite3";
import {FastifyReply, FastifyRequest} from "fastify";
import {z} from "zod";
import sanitizeHtml from "sanitize-html";
import bcrypt from "bcrypt";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

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
    name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
    email: z.string().email("Email invalide"),
    password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
});


export async function createClient(req: FastifyRequest, res: FastifyReply) {
    // const { name, email, password } = req.body as { name: string; email: string; password: string };
    const mystuff = req.body;
    const zod_result = clientSchema.safeParse(mystuff);
    console.log("GGGGGGGGGGGGGGGGGGGGGGGGGGGGG");
    if (!zod_result.success) {
        return zod_result.error.format();
    }
    const { data } = zod_result;
    console.log("GGGGGGGGGGGGGGGGG1GGGGGGGGGGG");
    let {name, email, password} = data;
    name = sanitizeHtml(name); // Protection from XSS attacks
    email = sanitizeHtml(email);
    password = sanitizeHtml(password);
    if (!name || !email || !password) {
        return res.status(454).send({error: "All information are required !"});
    }
    console.log("GGGGGGGGGGGGGGGGGG2GGGGGGGGGGG");
    const hashedPassword = await bcrypt.hash(password, 10);

    const insert = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)");
    const result = insert.run(name, email, hashedPassword);
    const rows = Client_db.prepare(`SELECT * FROM clients`).all();
    console.log("GGGGGGGGGGGGGGGG44444GGGGGGGGGGGGG");
    console.table(rows);
    if (result.changes === 1) {
        // const table = Client_db.prepare("SELECT * FROM clients");
        // console.table(table.database);
        Client_db.prepare("SELECT * FROM Client");
        const newClientId = result.lastInsertRowid;
        const newClient = Client_db.prepare("SELECT name FROM Client WHERE id = ?").get(newClientId) as { name: string } | undefined;
        if (newClient) {
            console.log("Nouveau client enregistré:", newClient.name);
        }
        return res.redirect(`/part/login?name=${encodeURIComponent(newClient?.name || "Utilisateur")}`);
        // return reply.redirect(`/part/login`);
        // return reply.status(201).send({ message: "Compte client créé avec succès", id: result.lastInsertRowid });
    } else {
        return res.status(500).send({ error: "Erreur lors de la création du compte" });
    }
}