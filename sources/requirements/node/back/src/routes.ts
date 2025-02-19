// routes.ts
// !!
// Separer en plusieurs fichier les types de route
// !!
import { FastifyInstance } from 'fastify';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import { env } from './env.js';
import { initDb, createClient } from './db.js'; // Import des fonctions pour la DB

export async function routes(fastify: FastifyInstance) {
    const db = await initDb(); // Initialisation de la base de données SQLite
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");

    // Route pour la page d'accueil
    fastify.get("/", (req, res) => {
        const file = readFileSync(pagePath, 'utf8');
        res.raw.writeHead(200, {'Content-Type': 'text/html'});
        res.raw.write(file);
        res.raw.end();
    });

    // Route pour enregistrer un client
    fastify.post('/register', async (req, res) => {
        const zParams = z.object({
            name: z.string().min(1),
            email: z.string().email(),
            password: z.string().min(6), // Minimum 6 caractères pour le mot de passe
        });
        const {success, error, data} = zParams.safeParse(req.body);

        if (!success) {
            res.raw.writeHead(400);
            res.raw.write(error);
            res.raw.end();
            return;
        }

        const {name, email, password} = data;

        try {
            await createClient(db, {name, email, password}); // Enregistrement dans la DB
            res.redirect('/register', 303); // Redirige vers la page d'inscription ou une autre page après inscription
        } catch (err) {
            res.raw.writeHead(500);
            res.raw.write('Erreur lors de l\'enregistrement');
            res.raw.end();
        }

    })

    // Route dynamique pour d'autres pages
    fastify.get("/:file", (req, res) => {
        const zParams = z.object({
            file: z.string(),
        });
        const {success, error, data} = zParams.safeParse(req.params);
        if (!success) {
            res.raw.writeHead(400);
            res.raw.write(error);
            res.raw.end();
            return;
        }
        let {file} = data;
        if (file === "front.js") {
            const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
            const file = readFileSync(frontPath, 'utf8');
            res.raw.writeHead(200, {'Content-Type': 'text/javascript'});
            res.raw.write(file);
            res.raw.end();
            return;
        }
        if (file.split('.').length === 1) {
            file = `${file}.html`;
        }
        const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, file);
        if (!existsSync(pagePath)) {
            res.raw.writeHead(404);
            res.raw.end();
            return;
        }
        const readFile = readFileSync(pagePath, 'utf8');
        res.raw.writeHead(200, {'Content-Type': 'text/html'});
        res.raw.write(readFile);
        res.raw.end();
    });
}