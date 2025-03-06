import { routes } from './routes.js';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import {join} from "node:path";
import {env} from "./env.js";
import {readFileSync} from "node:fs";
import {Client_db} from "./database.js";
import bcrypt from "bcrypt";
import {z} from "zod";
import sanitizeHtml from 'sanitize-html';

// Load SSL certificates
const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../src/static/secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../src/static/secure/cert.pem'))     // Certificate
    },
    logger: true
};
const fastify = Fastify(httpsOptions);

// Reset database and id to 1
// DELETE FROM clients;
// VACUUM;
// DELETE FROM clients;
// DELETE FROM sqlite_sequence;

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);

console.log('Base de données initialisée avec succès.');

const clientSchema = z.object({
    name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
    email: z.string().email("Email invalide"),
    password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
});

fastify.post("/post/create", async (request, reply) => {
    // Client_db.prepare(`DELETE FROM sqlite_sequence WHERE name = ?`).run("clients");
    try {
        const mystuff = request.body;
        const {success, data} =  clientSchema.safeParse(mystuff);
        if (!success) {
            return reply.status(400).send({ error: "Zod failed safeParse()" });
        }
        let {name, email, password} = data;
        // Vérifier que toutes les informations sont présentes
        console.log("============++++++++++++===============")
        console.log(name, email, password);
        name = sanitizeHtml(name); // Protection from XSS attacks
        email = sanitizeHtml(email);
        password = sanitizeHtml(password);
        console.log(name, email, password);
        console.log("============++++++++++++===============")
        if (!name || !email || !password) {
            return reply.status(400).send({ error: "Toutes les informations sont requises !" });
        }
        // Vérifier si l'email est déjà utilisé
        const existingClient = Client_db.prepare("SELECT * FROM clients WHERE email = ?").get(email);
        if (existingClient) {
            return reply.status(400).send({ error: "Email déjà utilisé" });
        }
        // Hasher le mot de passe
        const hashedPassword = await bcrypt.hash(password, 10);
        // Insérer le client dans la base de données
        const insert = Client_db.prepare("INSERT INTO clients (name, email, password) VALUES (?, ?, ?)");
        const result = insert.run(name, email, hashedPassword);
        // const rows = Client_db.prepare(`SELECT * FROM clients`).all();
        // console.table(rows);
        if (result.changes === 1) {
            const table = Client_db.prepare("SELECT * FROM clients");
            // console.log(table);
            const newClientId = result.lastInsertRowid;
            const newClient = Client_db.prepare("SELECT name FROM clients WHERE id = ?").get(newClientId) as { name: string } | undefined;
            if (newClient) {
                console.log("Nouveau client enregistré:", newClient.name);
                return reply.status(201).send({ redirect: `/part/login?name=${encodeURIComponent(newClient?.name || "Utilisateur")}` });
            }
            // return reply.redirect(`/part/login`);
            // return reply.status(201).send({ message: "Compte client créé avec succès", id: result.lastInsertRowid });
        } else {
            return reply.status(500).send({ error: "Erreur lors de la création du compte" });
        }
    } catch (error) {
        console.error("Erreur lors de l'inscription:", error);
        return reply.status(500).send({ error: "Erreur serveur" });
    }
});

fastify.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "src", "static"),
    prefix: "/static/"
})

fastify.get("/*", (req, res) => { // Route pour la page d'accueil
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.type('text/html').send(readFile);
});


// fastify.setNotFoundHandler((request, reply) => {
//     const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
//     const readFile = readFileSync(pagePath, 'utf8');
//     reply.type('text/html').send(readFile);
// });

fastify.register(routes)

fastify.listen({ host: '127.0.0.1', port: 3001 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})
