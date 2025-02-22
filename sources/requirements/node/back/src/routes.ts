// routes.ts
// !!
// Separer en plusieurs fichier les types de route
// !!
import { FastifyInstance } from 'fastify';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import { env } from './env.js';
import { CreateClient } from './database.js'; // Import des fonctions pour la DB

export async function routes(fastify: FastifyInstance) {
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");

    // Route pour la page d'accueil
    fastify.get("/", (req, res) => {
        env.LAST_URL = req.url;
        const file = readFileSync(pagePath, 'utf8');
        res.type('text/html').send(file);
    });

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
            env.LAST_URL = req.url;
            const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
            const file = readFileSync(frontPath, 'utf8');
            res.type('text/html').send(file);
            return;
        }
        if (file.split('.').length === 1) {
            file = `${file}.html`;
        }
        const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, file);
        const pathhome = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
        if (!existsSync(pagePath) || !existsSync(pathhome)) {
            res.raw.writeHead(404);
            res.raw.end();
            return;
        }
        if (env.LAST_URL === req.url) {
            const readFile = readFileSync(pagePath, 'utf8');
            const home = readFileSync(pathhome, 'utf8');
            const html = home.replace("<p>Choisissez une option pour charger le contenu.</p>", readFile);
            res.type('text/html').send(html);
            return;
        }
        // if client click on button
        env.LAST_URL = req.url;
        const readFile = readFileSync(pagePath, 'utf8');
        res.type('text/html').send(readFile);
    });

    // Route pour enregistrer un client
    fastify.post('/register', CreateClient);
}