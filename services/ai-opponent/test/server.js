"use strict";
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.ts                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/08 09:53:35 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/09 11:25:03 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
Object.defineProperty(exports, "__esModule", { value: true });
var fastify_1 = require("fastify");
var websocket_1 = require("@fastify/websocket");
var fastify = (0, fastify_1.default)({ logger: true });
var wss = (0, websocket_1.default)({ port: 8080 });
fastify.register(websocket_1.default);
wss.on('connection', function (ws) {
    console.log('Client Connected');
    ws.on('message', function (message) {
        var aligned = Buffer.from(message);
        var doubleArray = new Float64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
        console.log("Received:  ", Array.from(doubleArray));
        var float64 = new Float64Array(3);
        float64[0] = 0.2;
        float64[1] = 21.2;
        float64[2] = 3243.3;
        ws.send(float64);
    });
    ws.on('close', function () {
        console.log('Client disconnected');
    });
});
console.log('WebSocket server running on ws://localhost:8080');
