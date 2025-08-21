#include <regi.h>
#include <console.h>
#include <dfsel.h>
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/native_enum.h>
namespace py = pybind11;
using namespace regi;

// PYBIND11_MAKE_OPAQUE(std::vector<Card>)
// PYBIND11_MAKE_OPAQUE(std::vector<Combo>)

void bind_enums(pybind11::object &m)
{
    py::native_enum<Suit>(m, "Suit", "enum.IntEnum")
        .value("GLITCH", Suit::GLITCH)
        .value("CLUBS", Suit::CLUBS)
        .value("DIAMONDS", Suit::DIAMONDS)
        .value("HEARTS", Suit::HEARTS)
        .value("SPADES", Suit::SPADES)
        .export_values()
        .finalize();

    py::native_enum<Entry>(m, "Entry", "enum.IntEnum")
        .value("JOKER", Entry::JOKER)
        .value("ACE", Entry::ACE)
        .value("TWO", Entry::TWO)
        .value("THREE", Entry::THREE)
        .value("FOUR", Entry::FOUR)
        .value("FIVE", Entry::FIVE)
        .value("SIX", Entry::SIX)
        .value("SEVEN", Entry::SEVEN)
        .value("EIGHT", Entry::EIGHT)
        .value("NINE", Entry::NINE)
        .value("TEN", Entry::TEN)
        .value("JACK", Entry::JACK)
        .value("QUEEN", Entry::QUEEN)
        .value("KING", Entry::KING)
        .export_values()
        .finalize();

    py::native_enum<Powers>(m, "SuitPower", "enum.Flag")
        .value("DOUBLE_DAMAGE", Powers::CLUBS_DOUBLE)
        .value("DRAW_CARDS", Powers::DIAMONDS_DRAW)
        .value("REPLENISH", Powers::HEARTS_REPLENISH)
        .value("BLOCK", Powers::SPADES_BLOCK)
        .value("NERF", Powers::JOKER_NERF)
        .export_values()
        .finalize();

    py::native_enum<GameStatus>(m, "GameStatus", "enum.IntEnum")
        .value("LOADING", GameStatus::LOADING)
        .value("RUNNING", GameStatus::RUNNING)
        .value("ENDED", GameStatus::ENDED)
        .export_values()
        .finalize();

    py::native_enum<EndGameReason>(m, "EndGameReason", "enum.IntEnum")
        .value("INVALID_START", EndGameReason::INVALID_START)
        .value("NO_ENEMIES", EndGameReason::NO_ENEMIES)
        .value("BLOCK_FAILED", EndGameReason::BLOCK_FAILED)
        .value("ATTACK_FAILED", EndGameReason::ATTACK_FAILED)
        .value("PLAYER_DEAD", EndGameReason::PLAYER_DEAD)
        .export_values()
        .finalize();
}

void bind_cards(pybind11::object &m)
{
    py::class_<Card>(m, "Card")
        .def_property_readonly("entry", &Card::entry)
        .def_property_readonly("suit", &Card::suit)
        .def_property_readonly("strength", &Card::strength)
        .def(
            "__eq__", [](const Card &c1, const Card &c2) { return c1 == c2; },
            py::is_operator())
        .def("__str__",
             [](const Card &c)
             {
                 std::stringstream ss;
                 ss << c;
                 return ss.str();
             });

    py::class_<Enemy>(m, "Enemy")
        .def_readonly("HP", &Enemy::hp)
        .def_property_readonly("entry", &Enemy::entry)
        .def_property_readonly("suit", &Enemy::suit)
        .def_property_readonly("strength", &Enemy::strength)
        .def("__str__",
             [](const Enemy &e)
             {
                 std::stringstream ss;
                 ss << e;
                 return ss.str();
             });

    py::class_<Combo>(m, "Combo")
        .def_readonly("parts", &Combo::parts)
        .def_property_readonly("can_attack",
                               [](Combo &c) { return c.valid(true) != 0; })
        .def_property_readonly("base_damage", &Combo::getBaseDamage)
        .def_property_readonly("base_defense", &Combo::getBaseDefense)
        .def("__str__",
             [](const Combo &c)
             {
                 std::stringstream ss;
                 ss << c;
                 return ss.str();
             });
}

class PyBaseStrategy : public Strategy, py::trampoline_self_life_support
{
    /* TODO: Is there no other way? */
   public:
    using Strategy::Strategy;
    i32 setup(const Player &player, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, setup, player, g);
    }
    i32 getDefenseIndex(const std::vector<Combo> &combos, const Player &player,
                        i32 damage, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, getDefenseIndex, combos, player, damage, g);
    }
    i32 getAttackIndex(const std::vector<Combo> &combos, const Player &player,
                       bool yieldAllowed, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, getAttackIndex, combos, player, yieldAllowed, g);
    }
};

class PyRandomStrategy : public RandomStrategy, py::trampoline_self_life_support
{
    using RandomStrategy::RandomStrategy;
};

void bind_strat(pybind11::object &m)
{
    py::class_<Strategy, PyBaseStrategy /* trampoline */, py::smart_holder> base(
        m, "BaseStrategy");
    base.def(py::init<>())
        .def("setup", &Strategy::setup)
        .def("getAttackIndex", &Strategy::getAttackIndex)
        .def("getDefenseIndex", &Strategy::getDefenseIndex);
    py::class_<RandomStrategy, PyRandomStrategy, py::smart_holder>(m, "RandomStrategy",
                                                                   base)
        .def(py::init<>())
        .def("setup", &RandomStrategy::setup)
        .def("getAttackIndex", &RandomStrategy::getAttackIndex)
        .def("getDefenseIndex", &RandomStrategy::getDefenseIndex);
}

void bind_player(pybind11::object &m)
{
    py::class_<Player>(m, "Player")
        .def_readonly("cards", &Player::cards)
        .def_readonly("ID", &Player::id)
        .def_readonly("alive", &Player::alive)
        .def("__str__",
             [](const Player &player)
             {
                 std::stringstream ss;
                 ss << player;
                 return ss.str();
             });
}

class PyBaseLog : public BaseLog, py::trampoline_self_life_support
{
    /* TODO: Is there no other way? */
   public:
    using BaseLog::BaseLog;
    void attack(const Player &player, const Enemy &enemy, const Combo &combo,
                const i32 damage) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, attack, player, enemy, combo, damage);
    }
    void defend(const Player &player, const Combo &combo, const i32 damage) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, defend, player, combo, damage);
    }
    void failBlock(const Player &player, const i32 damage, const i32 maxblock) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, failBlock, player, damage, maxblock);
    }
    void drawOne(const Player &player) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, drawOne, player);
    }
    void replenish(const i32 n) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, replenish, n);
    }
    void enemyKill(const Enemy &enemy, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, enemyKill, enemy, g);
    }
    void state(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, state, g);
    }
    void debug(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, debug, g);
    }
    void endTurn(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, endTurn, g);
    }
    void startgame(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, startgame, g);
    }
    void endgame(EndGameReason e, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, endgame, e, g);
    }
    void postgame(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, postgame, g);
    }
};

void bind_log(pybind11::object &m)
{
    py::class_<BaseLog, PyBaseLog /* trampoline */, py::smart_holder>(m, "BaseLog")
        .def(py::init<>())
        .def("attack", &BaseLog::attack)
        .def("defend", &BaseLog::defend)
        .def("failBlock", &BaseLog::failBlock)
        .def("drawOne", &BaseLog::drawOne)
        .def("replenish", &BaseLog::replenish)
        .def("enemyKill", &BaseLog::enemyKill)
        .def("state", &BaseLog::state)
        .def("debug", &BaseLog::debug)
        .def("endTurn", &BaseLog::endTurn)
        .def("startgame", &BaseLog::startgame)
        .def("endgame", &BaseLog::endgame)
        .def("postgame", &BaseLog::postgame);
    /* TODO: why isn't this the same as RandomStrategy? */
    py::class_<ConsoleLog>(m, "CXXConsoleLog").def(py::init<>());
}

void bind_gamestate(pybind11::object &m)
{
    py::class_<GameState>(m, "GameState")
        .def(py::init([](BaseLog &log) { return GameState(log); }),
             py::keep_alive<1, 2>())
        .def(py::init([](ConsoleLog &log) { return GameState(log); }),
             py::keep_alive<1, 2>())
        .def("add_player", &GameState::addPlayer, py::keep_alive<1, 2>())
        .def_property_readonly("total_players", &GameState::totalPlayers)
        .def_property_readonly("hand_size", &GameState::getHandSize)
        .def_readonly("past_yields", &GameState::pastYieldsInARow)
        .def_readonly("status", &GameState::status)
        .def_readonly("draw_pile", &GameState::drawPile)
        .def_readonly("discard_pile", &GameState::discardPile)
        .def_readonly("enemy_pile", &GameState::enemyPile)
        .def_readonly("used_combos", &GameState::usedPile)
        .def("initialize",
             [](GameState &g)
             {
                 g.init();
                 g.setup();
                 return g.status;
             })
        .def("start_loop", &GameState::startLoop);
}

PYBIND11_MODULE(core, m)
{
    m.doc() = "c++ module for regicide game mechanics";
    bind_enums(m);
    bind_cards(m);
    bind_strat(m);
    bind_player(m);
    bind_log(m);
    bind_gamestate(m);
}
