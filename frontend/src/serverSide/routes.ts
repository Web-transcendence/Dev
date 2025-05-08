import { FastifyInstance } from 'fastify';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { env } from './env.js';

export async function routes(fastify: FastifyInstance) {
    fastify.get("/front.js", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    fastify.get("/pong.js", (req, res) => {
        const pongPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "pong.js");
        const file = readFileSync(pongPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    // ROAD OF TAG
    const htmlRoutes = [
        "about",
        "connected",
        "logout",
        "login",
        "register",
        "connect",
        "contact",
        "profile",
        "home",
        "towerRemote",
        "towerMode",
        "pongRemote",
        "pongLocal",
        "pongWatch",
        "pongMode",
        "tournaments",
        "lobby",
        "factor",
        "matchHistory",
    ];

    // Route dynamique pour chaque page définie
    htmlRoutes.forEach(route => {
        fastify.get(`/part/${route}`, (req, reply) => {
            const filename = route === "logout" ? "home.html" : `${route}.html`;
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, filename);
                console.log('FP: ', fullPath);
                const html = readFileSync(fullPath, "utf8");
                reply.type("text/html").send(html);
            } catch (error) {
                reply.code(404).send(`Page ${filename} non trouvée`);
            }
        });
    });

        // Route par défaut : redirige toute autre page sur index.html
        fastify.get('/part/:name', (req, reply) => {
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, 'index.html');
                const html = readFileSync(fullPath, 'utf8');
                reply.type('text/html').send(html);
            } catch (error) {
                reply.code(500).send('Erreur serveur : index.html introuvable');
            }
        });

    // FAV ICON
    fastify.get('/images/favicon.ico', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "nobglogo.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/images/login.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "login.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/images/BigLock.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "BigLock.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/images/logout.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "logout.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    // Png For Pong
    fastify.get('/assets/pong/ballup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/ballup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier ballup non trouvé");
        }
    });
    fastify.get('/assets/pong/bardown.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/bardown.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier bardown non trouvé");
        }
    });
    fastify.get('/assets/pong/barup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/barup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier barup non trouvé");
        }
    });
    fastify.get('/assets/pong/pong.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/pong.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier pong non trouvé");
        }
    });
    fastify.get('/assets/docker_compose.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "docker_compose.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier docker_compose non trouvé");
        }
    });
    fastify.get('/assets/tower-defense/:filename', function (req, reply) {
        try {
            const { filename } = req.params;
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'tower-defense', filename);
            const file = readFileSync(frontPath);
            reply.code(224).type('image/png').send(file);
        } catch (error) {
            reply.code(404).send(`Fichier ${req.params.filename} non trouvé`);
        }
    });
    //Menu mode pong
    fastify.get('/images/vsia.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "vsia.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier vsia non trouvé");
        }
    });
    fastify.get('/images/remote.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "remote.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier remote non trouvé");
        }
    });
    fastify.get('/images/solo.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "solo.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier solo non trouvé");
        }
    });
}