// websocket functions
function init_ws() {
    let g = Alpine.store('gamestate');
    let serverIP = window.location.host.toString();
    g.ws = new WebSocket(`ws://${serverIP}/ws/${g.userid}`);
    g.ws.onmessage = receive_ws;
    g.ws.onopen = function() {
        request_start();
    };
}

function send_ws(message)
{
    let g = Alpine.store('gamestate');
    let raw = JSON.stringify(message);
    let res = g.ws.send(raw);
}

function clearGameCookies() {
    document.cookie = "userid=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    document.cookie = "username=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
}

let _msgQueue = [];
let _msgProcessing = false;
const MSG_DELAY = 250;

function _processQueue() {
    if (_msgProcessing || _msgQueue.length === 0) return;
    _msgProcessing = true;
    let info = _msgQueue.shift();
    _handleMessage(info);
    _msgTimer = setTimeout(() => {
        _msgProcessing = false;
        _processQueue();
    }, MSG_DELAY);
}

function _flushQueue() {
    clearTimeout(_msgTimer);
    _msgProcessing = false;
    while (_msgQueue.length > 0) {
        _handleMessage(_msgQueue.shift());
    }
}

let _msgTimer = null;

function receive_ws(event) {
    // twice because string inside string
    let info = JSON.parse(JSON.parse(event.data));

    if (info.type === "invalid-session") {
        clearGameCookies();
        window.location.href = "/";
        return;
    }

    learnUsernames(info);
    if (info.type === "log") {
        _msgQueue.push(info);
        _processQueue();
    } else {
        // Flush pending log messages immediately before handling action messages
        _flushQueue();
        _handleMessage(info);
    }
}

function _handleMessage(info) {
    let g = Alpine.store('gamestate');
    let mainBtn = document.getElementById('main-button');
    let sideBtn = document.getElementById('side-button');
    if (info.data !== null) {
        console.log(info);
    }
    switch(info.type) {
        case "loading":
            logMessage(`${info.remain} players still need to connect`, 'is-secondary');
            g.statusz = "LOADING";
            addNotification('Waiting for other players to connect...', 'is-secondary');
            setButtonActivity(mainBtn, false);
            setButtonActivity(sideBtn, false);
            break;
        case "ready":
            g.statusz = "READY";
            setButtonActivity(mainBtn, true);
            setButtonActivity(sideBtn, false);
            break;
        case "log":
            processLog(info.data);
            g.history.push(info.data);
            break;
        case "select-attack":
            addNotification("Select cards for your attack.", 'is-primary');
            selectAttack(info.data);
            break;
        case "select-defend":
            addNotification("Select cards for your defense.", 'is-primary');
            selectDefend(info.data);
            break;
        case "select-redirect":
            logMessage(`select who plays next`, 'is-primary');
            selectRedirect(info.data);
            break;
        default:
            logMessage("fug", 'is-danger');
            logMessage(JSON.stringify(info));
            break;
    }
}

//
function reset_game() {
    // console.log("Resetting game");
    let g = Alpine.store('gamestate');
    document.getElementById('messages').replaceChildren();
    g.history = [];
    let message = {userid: g.userid, type:'player-reset', choice:0};
    send_ws(message);
}

function download_json() {
    let g = Alpine.store('gamestate');
    let text = JSON.stringify(g.history);
    let filename = `regi-${Date.now().toString()}.json`;
    let element = document.createElement('a');
    element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.click();
}

function request_start() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, username: g.username, type: "player-join", choice:0});
}
function player_ready() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, type: "player-ready", choice:0});
}




function submit_option() {
    let g = Alpine.store("gamestate");
    if (g.redirection) {
        submit_redirect_option();
    } else {
        submit_card_option();
    }
}

function submit_redirect_option() {
    let g = Alpine.store("gamestate");
    let players = document.getElementById('other-players').firstChild;
    let picked = [];
    for (let player of players.children) {
        if (player.classList.contains("is-focused")) {
            picked.push({ id: parseInt(player.dataset.playerId), name: player.innerHTML });
        }
    }

    let pickIndex = -1;
    if (picked.length != 1) {
        logMessage(`Invalid redirection!`, 'is-warning');
        addNotification("Invalid Redirection! Pick Again", 'is-warning');
    } else {
        pickIndex = picked[0].id;
        // send the option picked
        let msg = { userid: g.userid, type: "player-move", choice: pickIndex };
        send_ws(msg);

        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

        g.redirection = false;
    }
}

function submit_card_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    // get selected cards
    let cards = document.getElementById('player-cards').firstChild;
    let pickset = new Set([]);
    for (let card of cards.children) {
        if (card.classList.contains("is-focused")) {
            pickset.add(card.innerHTML);
        }
    }

    let picksOK = false;
    // console.log(cur_player);
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        let combset = new Set(cur_player.combos[i].map(x => x.value));
        if (pickset.isSubsetOf(combset) && combset.isSubsetOf(pickset)) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }

    if (!picksOK) {
        logMessage(`${Array.from(pickset)} is not a valid move!`, 'is-warning');
        updateCards(g.data.player);
        addNotification("Invalid Move! Pick Cards Again", 'is-warning');
    } else {
        // send the option picked
        let msg = { userid: g.userid, type: "player-move", choice: pickIndex };
        send_ws(msg);

        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

    }
}

function yield_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    
    let picksOK = false;
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        if (combos[i].length == 0) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }
    
    if (!picksOK) {
        logMessage("yield is not a valid move!", 'is-warning');
        addNotification("Invalid Yield! Pick Cards", 'is-warning');
    } else {
        // send the option picked
        let msg = { userid: g.userid, choice: pickIndex };
        send_ws(msg);

        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

    }
}

function setButtonActivity(button, active) {
    if (active) {
        button.classList.remove("is-loading")
    } else {
        button.classList.add("is-loading")
    }
    if (button.id === 'side-button') {
        Alpine.store('gamestate').sideButtonActive = active;
    }
}

function makeEnemyPileWithInfo(ce, enemyPileSize) {
    let wrapper = document.createElement("div");
    wrapper.className = "card-wrapper";

    let depth = enemyPileSize > 0 ? (enemyPileSize / 12) * 0.5 : 0;

    let pile = document.createElement("div");
    pile.className = "box has-background-danger-light game-card game-card-enemy";
    pile.style.boxShadow = depth > 0
        ? `${depth}rem ${depth}rem 0 rgba(0,0,0,0.3), ${depth}rem ${depth}rem 0 -0.0625rem rgba(0,0,0,0.1)`
        : "none";
    pile.style.marginBottom = depth + "rem";

    let name = document.createElement("div");
    name.className = "has-text-weight-bold is-size-5";
    name.textContent = ce.value;
    pile.appendChild(name);

    let hp = document.createElement("div");
    hp.className = "is-size-6";
    hp.textContent = `${ce.hp} HP`;
    pile.appendChild(hp);

    let lbl = document.createElement("div");
    lbl.className = "is-size-7 label-below";
    lbl.textContent = `${enemyPileSize} enemies left`;

    wrapper.appendChild(pile);
    wrapper.appendChild(lbl);
    return wrapper;
}

function makePileVisual(label, count, maxCount, colorClass) {
    let wrapper = document.createElement("div");
    wrapper.className = "pile-wrapper";

    // Shadow depth based on card count (0-0.5rem)
    let depth = maxCount > 0 ? (count / maxCount) * 0.5 : 0;

    let pile = document.createElement("div");
    pile.className = `box ${colorClass} game-card`;
    pile.style.boxShadow = depth > 0
        ? `${depth}rem ${depth}rem 0 rgba(0,0,0,0.3), ${depth}rem ${depth}rem 0 -0.0625rem rgba(0,0,0,0.1)`
        : "none";
    pile.style.marginBottom = depth + "rem";

    let num = document.createElement("span");
    num.className = "is-size-3 has-text-weight-bold";
    num.textContent = count;
    pile.appendChild(num);

    let lbl = document.createElement("div");
    lbl.className = "is-size-7 label-below";
    lbl.textContent = label;

    wrapper.appendChild(pile);
    wrapper.appendChild(lbl);
    return wrapper;
}

function makePilesRow(game) {
    let res = document.createElement("div");
    res.className = "mb-4";

    let row = document.createElement("div");
    row.className = "pile-row";
    let totalCards = game.draw_pile_size + game.discard_pile_size;
    row.appendChild(makePileVisual("Draw", game.draw_pile_size, totalCards || 42, "has-background-info-light"));
    row.appendChild(makePileVisual("Discard", game.discard_pile_size, totalCards || 42, "has-background-warning-light"));
    res.appendChild(row);
    return res;
}

function makeOtherPlayerInfo(game) {
    let res = document.createElement("div");
    res.className = "mb-4";
    let hdr = document.createElement("h2");
    hdr.className = "subtitle is-5";
    hdr.textContent = "Players";
    res.appendChild(hdr);

    let table = document.createElement("table");
    table.className = "table is-narrow is-size-6";
    for (let player of game.players) {
        let row = document.createElement("tr");
        if (player.id === game.active_player_id) {
            row.classList.add("current-player");
        }
        let nameCell = document.createElement("td");
        nameCell.textContent = getDisplayName(player);
        let cardCell = document.createElement("td");
        cardCell.className = "has-text-right";
        cardCell.textContent = `${player.num_cards} cards`;
        row.appendChild(nameCell);
        row.appendChild(cardCell);
        table.appendChild(row);
    }
    res.appendChild(table);
    return res;
}

function makeUsedCombos(combos) {
    let res = document.createElement("div");
    res.className = "mb-4";
    let hdr = document.createElement("h2");
    hdr.className = "subtitle is-5";
    hdr.textContent = "Combos Used";
    res.appendChild(hdr);

    let list = document.createElement("div");
    list.className = "combos-list";
    for (let combo of combos) {
        let b = document.createElement("div");
        b.className = "combo-card has-text-weight-bold is-size-7";
        for (let card of combo) {
            let val = document.createElement("div");
            val.textContent = card.value;
            b.appendChild(val);
        }
        list.appendChild(b);
    }
    res.appendChild(list);
    return res;
}

function updateBoard(game) {
    let piles_view = document.getElementById('piles-view');
    piles_view.replaceChildren();
    piles_view.appendChild(makePilesRow(game));

    let players_view = document.getElementById('players-view');
    players_view.replaceChildren();
    players_view.appendChild(makeOtherPlayerInfo(game));

    let enemy_view = document.getElementById('enemy-view');
    enemy_view.replaceChildren();
    enemy_view.appendChild(makeEnemyPileWithInfo(game.current_enemy, game.enemy_pile_size));

    let combos_view = document.getElementById('combos-view');
    combos_view.replaceChildren();
    combos_view.appendChild(makeUsedCombos(game.used_combos || []));
}

function getCardButton(card, block) {
    let b = document.createElement("div");
    b.className = "game-card game-card-selectable has-text-weight-bold is-size-6";
    b.textContent = card;
    b.addEventListener("click", () => {
        b.classList.toggle("is-focused");
    });
    return b;
}

function getPlayerButton(player, block) {
    let b = document.createElement("div");
    b.className = "button is-link";
    b.addEventListener("click", () => { 
        if (b.classList.contains("is-focused")) {
            b.classList.remove("is-focused") 
            b.classList.remove("is-dark")
            b.classList.add("is-link")
        } else {
            // mutually exclusive
            for (const otherb of block.children) {
                otherb.classList.remove("is-focused") 
                otherb.classList.remove("is-dark")
                otherb.classList.add("is-link")
            }
            b.classList.add("is-focused")
            b.classList.add("is-dark")
            b.classList.remove("is-link")
        }
    });
    b.dataset.playerId = player.id;
    b.innerHTML = getDisplayName(player);
    return b;
}

function updateCards(player) {
    let g = Alpine.store('gamestate');
    let target = document.getElementById('player-cards');
    target.replaceChildren();

    let bgroup = document.createElement("div");
    bgroup.className = "cards-row";
    let cards = player.cards;
    for (let i = 0, len = cards.length; i < len; i++) {
        bgroup.appendChild(getCardButton(cards[i]));
    }
    target.appendChild(bgroup);
}
function updateOtherPlayers(player, game) {
    let g = Alpine.store('gamestate');
    let target = document.getElementById('other-players');
    target.replaceChildren();

    let bgroup = document.createElement("div");
    let players = game.players;
    bgroup.className = "buttons has-addons are-medium";
    for (let i = 0, len = players.length; i < len; i++) {
        if (players[i].id != player.id) {
            bgroup.appendChild(getPlayerButton(players[i], bgroup));
        }
    }
    target.appendChild(bgroup);
    // console.log(player);
}

function selectAttack(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = false;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, data.yield_allowed);
    //
    g.myTurn = true;
    g.playerShould = "You have to ATTACK";
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}
function selectDefend(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = false;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, false);
    //
    g.myTurn = true;
    g.playerShould = `You have to DEFEND ${data.damage} damage`;
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}
function selectRedirect(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = true;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, false);
    //
    g.myTurn = true;
    g.playerShould = `You have to pick the next player`;
    addNotification("Select who will play next after you", 'is-primary');
    // console.log(data);
    updateBoard(data.game);
    updateOtherPlayers(data.player, data.game);
}

function learnUsernames(data) {
    let g = Alpine.store('gamestate');
    function learnPlayer(p) {
        if (p && p.id !== undefined && p.username) {
            g.knownUsernames[p.id] = p.username;
        }
    }
    if (!data || typeof data !== 'object') return;
    learnPlayer(data.player);
    learnPlayer(data.active_player);
    if (Array.isArray(data.players)) {
        for (let p of data.players) learnPlayer(p);
    }
    if (data.game && typeof data.game === 'object') {
        learnPlayer(data.game.active_player);
        if (Array.isArray(data.game.players)) {
            for (let p of data.game.players) learnPlayer(p);
        }
    }
    if (data.data && typeof data.data === 'object') {
        learnUsernames(data.data);
    }
}

function getDisplayName(player) {
    let g = Alpine.store('gamestate');
    if (player.strategy === "player-webui") {
        let name = player.username || g.knownUsernames[player.id];
        if (name) return name;
    }
    if (player.strategy) {
        return `Bot ${player.id} (${player.strategy})`;
    }
    return player.username || g.knownUsernames[player.id] || `Bot ${player.id}`;
}

function updateCurrentPlayerCards(game) {
    let g = Alpine.store('gamestate');
    if (game && game.players) {
        let me = game.players.find(p => p.id == g.playerid);
        if (me && me.cards) {
            updateCards(me);
        }
    }
}

// logging
function processLog(data) {
    let g = Alpine.store('gamestate');
    if (data.game != null) {
        updateBoard(data.game);
        updateCurrentPlayerCards(data.game);
    }
    switch (data.event) {
        case 'STARTGAME':
            if (data.game != null && data.game.active_player_id !== null) {
                g.statusz = "RUNNING";
                logMessage("Game has started", 'is-primary');
            }
            break;
        case 'ATTACK':
            let combo1 = data.combo.map(x => x.value);
            logMessage(`${getDisplayName(data.player)} attacked ${data.enemy.value} with ${combo1}`, 'is-info');
            logMessage(`${getDisplayName(data.player)} dealt ${data.damage} damage!`, 'is-info');
            break;
        case 'DEFEND':
            let combo2 = data.combo.map(x => x.value);
            logMessage(`${data.enemy.value} attacked ${getDisplayName(data.player)} for ${data.damage} damage`, 'is-info');
            logMessage(`${getDisplayName(data.player)} blocked with ${combo2}`, 'is-info');
            break;
        case 'REDIRECT':
            let nextPlayerName = "Bot " + data.next_playerid;
            if (data.game && data.game.players) {
                const nextPlayer = data.game.players.find(p => p.id === data.next_playerid);
                if (nextPlayer) {
                    nextPlayerName = getDisplayName(nextPlayer);
                }
            }
            logMessage(`${getDisplayName(data.player)} redirected play to ${nextPlayerName}`, 'is-info');
            break;
        case 'ENEMYKILL':
            let exact = data.enemy.hp === 0 ? " exact " : "";
            logMessage(`${data.enemy.value} killed` + exact + "!", 'is-success'); 
            break;
        case 'REPLENISH':
            logMessage(`${data.n_cards} cards added back to the draw pile`, 'is-info');
            break;
        case 'DRAWONE':
            logMessage(`${getDisplayName(data.player)} drew a card`);
            break;
        case 'DECKEMPTY':
            logMessage(`${getDisplayName(data.player)} cannot draw`);
            break;
        case 'ENDGAME':
            endGameStatusUpdate();
            if (data.game && data.game.enemy_pile_size === 0 && (!data.game.current_enemy || data.game.current_enemy.hp <= 0)) {
                g.playerShould = "Victory! All enemies defeated!";
                logMessage("Victory! All enemies defeated!", 'is-success');
            } else {
                let endPlayer = data.player
                    || (data.game && data.game.active_player)
                    || (data.game && data.game.players && data.game.players.find(p => p.id === data.game.active_player_id));
                let endPlayerName = endPlayer ? getDisplayName(endPlayer) : ("Bot " + (data.game ? data.game.active_player_id : "?"));
                g.playerShould = `Defeat! ${endPlayerName} could not survive.`;
                logMessage("Game has ended - Defeat!", 'is-danger');
            }
            break;
        case 'POSTGAME':
            endGameStatusUpdate();
            logMessage("Game over!", 'is-primary');
            break;
        case 'FAILBLOCK':
            logMessage(`${data.enemy.value} attacked ${getDisplayName(data.player)} for ${data.damage}`, 'is-danger');
            logMessage(`${getDisplayName(data.player)} can block at most ${data.maxblock}!`, 'is-danger');
            g.playerShould = `${getDisplayName(data.player)} failed to block ${data.enemy.value}'s attack of ${data.damage} (max block: ${data.maxblock})`;
            break;
        case 'FULLBLOCK':
            logMessage(`${data.enemy.value} is blocked by ${getDisplayName(data.player)}`, 'is-info');
            break;
        case 'STATE':
            g.statusz = "RUNNING";
            if (data.game.active_player_id !== null) {
                if (data.game.active_player_id != g.playerid) {
                    g.myTurn = false;
                    let submitter = document.getElementById('main-button');
                    let yielder = document.getElementById('side-button');
                    setButtonActivity(submitter, false);
                    setButtonActivity(yielder, false);

                    let activePlayerName = data.game.active_player
                        ? getDisplayName(data.game.active_player)
                        : "Bot " + data.game.active_player_id;
                    g.playerShould = `Waiting for ${activePlayerName}...`;
                    logMessage(`Wait for ${activePlayerName} to play..`);
                } else {
                    updateCards(data.game.active_player);
                }
            }
        case 'DEBUG':
            // can I send something here?
            break;
        // default:
            // console.log(data.event);
    }
}

function setupTurnMessageHover() {
    let tray = document.getElementById('turn-message');
    if (tray.dataset.hoverSetup) return;
    tray.dataset.hoverSetup = 'true';

    tray.addEventListener('mouseenter', () => {
        tray.querySelectorAll('.notification').forEach(n => {
            clearTimeout(parseInt(n.dataset.fadeTimer));
            clearTimeout(parseInt(n.dataset.removeTimer));
            n.style.transition = '';
            n.style.opacity = '1';
        });
    });
    tray.addEventListener('mouseleave', () => {
        dismissOldNotifications(200);
    });
}

function restackNotifications() {
    // no-op: notifications are absolutely positioned via inline styles set on creation
}

function dismissOldNotifications(delay) {
    let tray = document.getElementById('turn-message');
    let notifications = tray.querySelectorAll('.notification');
    for (let i = 0; i < notifications.length; i++) {
        scheduleDismiss(notifications[i], delay + i * 300);
    }
}

function scheduleDismiss(el, delay) {
    let fadeTimer = setTimeout(() => {
        el.style.transition = 'opacity 3500ms ease-in-out';
        el.style.opacity = '0';
    }, delay);
    let removeTimer = setTimeout(() => {
        el.remove();
        restackNotifications();
    }, delay + 3600);
    el.dataset.fadeTimer = fadeTimer;
    el.dataset.removeTimer = removeTimer;
}

function addNotification(content, subtype) {
    let tray = document.getElementById('turn-message');
    setupTurnMessageHover();
    //
    let trayHeight = tray.offsetHeight;
    let inset = 12;
    let notifHeight = trayHeight - (inset * 2);

    let res = document.createElement('div');
    res.className = `notification is-light is-size-5 notif-overlay ${subtype}`;
    res.role = "alert";
    res.style.height = (notifHeight / 16) + 'rem';
    res.style.maxHeight = (notifHeight / 16) + 'rem';

    let delButton = document.createElement('button');
    delButton.classList.add('delete');
    delButton.addEventListener("click", () => {
        res.remove();
        restackNotifications();
    });

    res.appendChild(delButton);
    res.appendChild(document.createTextNode(content));

    // Ensure newest is always on top
    let existing = tray.querySelectorAll('.notification');
    existing.forEach(n => { n.style.zIndex = '1'; });
    res.style.zIndex = '10';
    res.classList.add('notif-slide-in');
    tray.appendChild(res);
    dismissOldNotifications(600);
}

function logMessage(content, subtype=null) {
    let g = Alpine.store('gamestate');
    let messages = document.getElementById('messages')
    let message = document.createElement('li')
    message.appendChild(document.createTextNode(content))
    messages.prepend(message)
    if (subtype != null) {
        setTimeout(() => {addNotification(content, subtype);}, 1000);
    }
}

// UI
function gameStatusShowable() {
    let g = Alpine.store('gamestate');
    let showStates = ["READY", "RUNNING", "ENDED"];
    return showStates.includes(g.statusz);
}

function endGameStatusUpdate() {
    let g = Alpine.store('gamestate');
    g.statusz = "ENDED";
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, true);
}

function mainButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
            result = "Waiting...";
            break;
        case "READY":
            result = "Ready!"
            break;
        case "RUNNING":
            result = "Submit";
            break;
        case "ENDED":
        case "ERROR":
            result = "Restart";
            break;
        default:
            result = "???";
    }
    return result;
}
function mainButtonRedirect() {
    let button = document.getElementById("main-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    let g = Alpine.store('gamestate');
    switch (g.statusz) {
        case "LOADING":
            break;
        case "READY":
            player_ready();
            break;
        case "RUNNING":
            submit_option();
            break;
        case "ENDED":
        case "ERROR":
            reset_game();
            break;
    }
}

function sideButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
        case "READY":
        case "ERROR":
            result = "<hide>";
            break;
        case "RUNNING":
            result = "Yield";
            break;
        case "ENDED":
            result = g.noDownload ? "<hide>" : "Download JSON";
            break;
        default:
            result = "???";
    }
    return result;
}

function sideButtonRedirect() {
    let g = Alpine.store('gamestate');
    let button = document.getElementById("side-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    switch (g.statusz) {
        case "LOADING":
            break;
        case "READY":
            break;
        case "RUNNING":
            yield_option();
            break;
        case "ENDED":
        case "ERROR":
            download_json();
            break;
    }
}


