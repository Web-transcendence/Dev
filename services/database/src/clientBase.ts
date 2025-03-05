import Database from "better-sqlite3";
export const Client_db = new Database('client.db')  // Importation correcte de sqlite
import {FastifyReply, FastifyRequest} from "fastify";


Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);

export function emailExist(email: string) {
    const existingClient = Client_db.prepare("SELECT * FROM clients WHERE email = ?").get(email);
    return !!existingClient;
}


export async function createClient(req: FastifyRequest, res: FastifyReply) {
    const { name, email, password } = req.body as { name: string; email: string; password: string };

    const insert = Client_db.prepare("INSERT INTO clients (name, email, password) VALUES (?, ?, ?)");
    const result = insert.run(name, email, password);
    const rows = Client_db.prepare(`SELECT * FROM clients`).all();
    console.table(rows);
    if (result.changes === 1) {
        const table = Client_db.prepare("SELECT * FROM clients");
        console.log(table);
        const newClientId = result.lastInsertRowid;
        const newClient = Client_db.prepare("SELECT name FROM clients WHERE id = ?").get(newClientId) as { name: string } | undefined;
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